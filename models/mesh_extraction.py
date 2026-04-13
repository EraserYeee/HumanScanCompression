"""Delaunay-based mesh extraction from predicted seed points + normals.

Adapted from PoNQ (CVPR 2024):
  - PoNQ/src/utils/PoNQ_to_mesh.py: MeshFromPoNQ class
  - PoNQ/src/utils/mesh_tools.py: tet_circumcenter
"""

import numpy as np
from scipy.spatial import Delaunay, QhullError
import torch


# 8 bounding-box corners: all sign combinations (from PoNQ mesh_tools.py:9-16)
_SIGNS = np.array(
    [[(-1) ** i, (-1) ** j, (-1) ** k]
     for i in range(2) for j in range(2) for k in range(2)],
    dtype=np.float64,
)


def tet_circumcenter(verts):
    """Compute circumcenter of tetrahedra.

    Adapted from PoNQ/src/utils/mesh_tools.py:280-327.

    Args:
        verts: (T, 4, 3) numpy array of tet vertex positions.

    Returns:
        (T, 3) circumcenter positions.
    """
    ba_x = verts[:, 1, 0] - verts[:, 0, 0]
    ba_y = verts[:, 1, 1] - verts[:, 0, 1]
    ba_z = verts[:, 1, 2] - verts[:, 0, 2]
    ca_x = verts[:, 2, 0] - verts[:, 0, 0]
    ca_y = verts[:, 2, 1] - verts[:, 0, 1]
    ca_z = verts[:, 2, 2] - verts[:, 0, 2]
    da_x = verts[:, 3, 0] - verts[:, 0, 0]
    da_y = verts[:, 3, 1] - verts[:, 0, 1]
    da_z = verts[:, 3, 2] - verts[:, 0, 2]

    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z

    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y

    div_den = ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z
    mask_coplanar = np.abs(div_den) < 1e-12
    div_den[mask_coplanar] = 1.0
    denominator = 0.5 / div_den

    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z) * denominator

    out = np.column_stack((circ_x, circ_y, circ_z)) + verts[:, 0]
    out[mask_coplanar] = verts[mask_coplanar].mean(axis=1)
    return out


class DelaunayMeshExtractor:
    """Extract a triangle mesh from predicted seed positions and normals.

    Adapted from PoNQ's MeshFromPoNQ (PoNQ_to_mesh.py).
    Non-differentiable topology computation; returns vertex indices that
    preserve gradient when used to index into the differentiable position tensor.

    Pipeline:
        1. Add bounding-box corner vertices
        2. scipy.spatial.Delaunay tetrahedralization
        3. Circumcenter-normal tet classification (inside/outside)
        4. Barycenter correction for ambiguous tets
        5. Surface extraction (boundary triangles between inside/outside)
    """

    def __init__(self, bbox_padding: float = 0.15, vote_threshold: int = 3):
        """
        Args:
            bbox_padding: Fractional padding added to bounding box for corner vertices.
            vote_threshold: Minimum normal votes (out of 4) to classify a tet as inside.
                            3 is conservative; 2 is more aggressive.
        """
        self.bbox_padding = bbox_padding
        self.vote_threshold = vote_threshold

    @torch.no_grad()
    def extract(self, positions: torch.Tensor, normals: torch.Tensor):
        """Extract triangle mesh from seed positions and normals.

        All computation is on CPU (numpy/scipy). Returns indices into the
        original positions tensor so that ``positions[vert_indices]`` preserves
        autograd connectivity.

        Args:
            positions: (N, 3) predicted seed positions (may have grad).
            normals:   (N, 3) predicted unit normals (may have grad).

        Returns:
            faces:        (F, 3) int64 tensor, face vertex indices into [0..V-1].
            vert_indices: (V,) int64 tensor, maps output vertices to original
                          seed indices [0..N-1]. Use as ``base_verts = positions[vert_indices]``.
        """
        N = positions.shape[0]
        pos_np = positions.detach().cpu().numpy().astype(np.float64)
        nor_np = normals.detach().cpu().numpy().astype(np.float64)

        # --- Step 1: Add bounding-box corners (PoNQ lines 23-30) ---
        bbox_min = pos_np.min(axis=0)
        bbox_max = pos_np.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_extent = (bbox_max - bbox_min) / 2.0 * (1.0 + self.bbox_padding)
        corners = bbox_center + bbox_extent * _SIGNS  # (8, 3)

        pos_ext = np.concatenate([pos_np, corners], axis=0)  # (N+8, 3)
        # Corner normals point inward (toward center)
        corner_normals = -_SIGNS / np.sqrt((_SIGNS ** 2).sum(-1, keepdims=True))
        nor_ext = np.concatenate([nor_np, corner_normals], axis=0)

        # --- Step 2: Delaunay tetrahedralization (PoNQ line 36) ---
        try:
            # Add small jitter to avoid degenerate configurations
            jitter = np.random.RandomState(42).randn(*pos_ext.shape) * 1e-7
            tri = Delaunay(pos_ext + jitter)
        except QhullError:
            # Fallback: return convex hull of original points
            return self._fallback_convex_hull(pos_np, N)

        simplices = tri.simplices  # (T, 4)
        neighbors = tri.neighbors  # (T, 4)

        # --- Step 3: Circumcenter-normal tet classification ---
        # (PoNQ get_init_tet_color, lines 106-111)
        circum_centers = tet_circumcenter(pos_ext[simplices])  # (T, 3)
        vects = pos_ext[simplices] - circum_centers[:, None, :]  # (T, 4, 3)
        votes = (vects * nor_ext[simplices]).sum(-1) > 0  # (T, 4) bool
        tet_colors = votes.sum(-1).astype(np.int32)  # (T,) in [0, 4]

        # Force tets containing corner vertices to outside
        has_corner = (simplices >= N).any(axis=-1)
        tet_colors[has_corner] = 0

        # --- Step 4: Barycenter correction (PoNQ correct_tet_color, lines 113-121) ---
        bary_centers = pos_ext[simplices].mean(axis=1)  # (T, 3)
        vects_from_bary = bary_centers[:, None, :] - pos_ext[simplices]  # (T, 4, 3)
        dist_to_plane = (vects_from_bary * nor_ext[simplices]).sum(-1)  # (T, 4)
        mean_dist = dist_to_plane.mean(axis=-1)  # (T,)

        is_prob_in = mean_dist < 0
        is_prob_out = mean_dist > 0
        tet_colors[(tet_colors == 0) & is_prob_in] = 1
        tet_colors[(tet_colors == 4) & is_prob_out] = 1

        # --- Step 5: Extract triangle faces (PoNQ get_triangle_faces, lines 55-63) ---
        opp_face = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        T = len(simplices)
        ii = np.arange(T)
        triangle_faces = np.zeros((T * 4, 3), dtype=np.int64)
        triangle_neighbors = np.zeros((T * 4, 2), dtype=np.int64)
        for j in range(4):
            triangle_faces[4 * ii + j] = simplices[:, opp_face[j]]
            triangle_neighbors[4 * ii + j, 0] = ii
            triangle_neighbors[4 * ii + j, 1] = neighbors[:, j]

        # --- Step 6: Orient faces consistently (PoNQ order_neighbors, lines 68-77) ---
        opp_vert_idx = np.array([3, 2, 1, 0])  # opposite vertex for each face in a tet
        opp_verts = simplices[:, opp_vert_idx].flatten()  # (T*4,)
        p1 = pos_ext[triangle_faces[:, 0]]
        p2 = pos_ext[triangle_faces[:, 1]]
        p3 = pos_ext[triangle_faces[:, 2]]
        vp = pos_ext[opp_verts]
        # Face normal should point away from opposite vertex
        cross = np.cross(p2 - p1, p3 - p1)
        dot = (cross * (vp - (p1 + p2 + p3) / 3.0)).sum(-1)
        needs_flip = dot > 0
        # Flip faces that point toward opposite vertex
        flipped = triangle_faces[:, [0, 2, 1]]
        triangle_faces = np.where(needs_flip[:, None], flipped, triangle_faces)

        # --- Step 7: Surface extraction (PoNQ get_surface, lines 185-214) ---
        # Binary classification: inside if votes >= threshold
        binary_inside = tet_colors >= self.vote_threshold

        # For each triangle face, get the colors of its two adjacent tets
        neigh_tet_a = triangle_neighbors[:, 0]  # (T*4,)
        neigh_tet_b = triangle_neighbors[:, 1]  # (T*4,)

        color_a = np.zeros(T * 4, dtype=bool)
        color_b = np.zeros(T * 4, dtype=bool)
        color_a = binary_inside[neigh_tet_a]

        valid_neigh = neigh_tet_b >= 0
        color_b[valid_neigh] = binary_inside[neigh_tet_b[valid_neigh]]
        # Boundary with infinity (neighbor == -1) counts as outside
        color_b[~valid_neigh] = False

        # Select faces at inside/outside boundary
        # One side inside (True), other side outside (False)
        boundary_mask = (color_a != color_b)
        surface_faces = triangle_faces[boundary_mask]

        # Ensure faces point outward (from inside toward outside)
        # If color_a is outside and color_b is inside, flip the face
        needs_outward_flip = (~color_a[boundary_mask]) & (color_b[boundary_mask])
        flipped_surface = surface_faces[:, [0, 2, 1]]
        surface_faces = np.where(needs_outward_flip[:, None], flipped_surface, surface_faces)

        if len(surface_faces) == 0:
            return self._fallback_convex_hull(pos_np, N)

        # --- Step 8: Remove corner vertices and re-index ---
        # Filter out any face that references a corner vertex
        has_corner_face = (surface_faces >= N).any(axis=-1)
        surface_faces = surface_faces[~has_corner_face]

        if len(surface_faces) == 0:
            return self._fallback_convex_hull(pos_np, N)

        # Filter degenerate faces (zero area)
        v0 = pos_np[surface_faces[:, 0]]
        v1 = pos_np[surface_faces[:, 1]]
        v2 = pos_np[surface_faces[:, 2]]
        areas = np.sqrt((np.cross(v1 - v0, v2 - v0) ** 2).sum(-1))
        valid_area = areas > 1e-8
        surface_faces = surface_faces[valid_area]

        if len(surface_faces) == 0:
            return self._fallback_convex_hull(pos_np, N)

        # Compact re-indexing: only keep vertices used by surface faces
        used_verts = np.unique(surface_faces)
        remap = np.full(N + 8, -1, dtype=np.int64)
        remap[used_verts] = np.arange(len(used_verts))
        compact_faces = remap[surface_faces]

        device = positions.device
        faces_tensor = torch.from_numpy(compact_faces).long().to(device)
        vert_indices = torch.from_numpy(used_verts).long().to(device)

        return faces_tensor, vert_indices

    def _fallback_convex_hull(self, pos_np, N):
        """Return convex hull as fallback when normal-based extraction fails."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pos_np)
            faces = hull.simplices.astype(np.int64)
            used = np.unique(faces)
            remap = np.full(N, -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            compact_faces = remap[faces]
            # ConvexHull faces may need orientation fix
            return (
                torch.from_numpy(compact_faces).long(),
                torch.from_numpy(used).long(),
            )
        except Exception:
            # Last resort: single degenerate triangle at first 3 points
            return (
                torch.tensor([[0, 1, 2]], dtype=torch.long),
                torch.tensor([0, 1, 2], dtype=torch.long),
            )
