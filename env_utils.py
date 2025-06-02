# Per-object forces forces that can be applied: left, right, up, down; action.
N_FORCES = 5
N_MOVEMENTS = 4
ACTION = 4


def multihot_to_desc(multihot_level, objs_to_idxs, n_objs, show_background=False):
    """Converts a multihot array to a 2D list of descriptions.
    
    Args:
        multihot_level: A multihot array of shape [n_objects + n_forces, height, width].
        obj_to_idxs: Dictionary mapping object names to their indices.
    
    Returns:
        A 2D list where each cell contains a string describing all objects and forces present.
    """
    height, width = multihot_level.shape[1:]
    
    # Create a reverse mapping from indices to object names
    idxs_to_obj = {idx: obj for obj, idx in objs_to_idxs.items()}
    
    # Create the description grid
    desc_grid = []
    for h in range(height):
        row = []
        for w in range(width):
            cell_desc = []
            
            # Check which objects are active in this cell
            for obj_idx in range(n_objs):
                if multihot_level[obj_idx, h, w] > 0:
                    obj_name = idxs_to_obj[obj_idx]
                    if obj_name == 'background' and not show_background:
                        continue
                    obj_desc = obj_name
                    
                    # Check if there's a force applied to this object
                    force_names = ["left", "down", "right", "up", "action"]
                    forces = []
                    for f_idx, force_name in enumerate(force_names):
                        force_channel = n_objs + (obj_idx * N_FORCES) + f_idx
                        if force_channel < multihot_level.shape[0] and multihot_level[force_channel, h, w] > 0:
                            forces.append(f"force {force_name}")
                    
                    if forces:
                        obj_desc += f" ({', '.join(forces)})"
                    
                    cell_desc.append(obj_desc)
            
            row.append(", ".join(cell_desc) if cell_desc else "empty")
        desc_grid.append(row)
    
    # Find the maximum width for each column
    column_widths = []
    for w in range(width):
        max_width = 0
        for h in range(height):
            max_width = max(max_width, len(desc_grid[h][w]))
        column_widths.append(max_width)
    
    # Add padding to each cell
    for h in range(height):
        for w in range(width):
            desc_grid[h][w] = desc_grid[h][w].ljust(column_widths[w])
    
    # Join rows with consistent spacing
    desc_str = "\n".join([" || ".join(row) for row in desc_grid])
    
    return desc_str
