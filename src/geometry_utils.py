import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from .simulation_wrapper import _generate_ssh_ring_geometry

def visualize_ring_geometry(design_vector, config, disorder_std=0.0, save_path=None):
    """
    Visualize the SSH ring resonator geometry.
    
    Parameters:
    - design_vector: [a, b, r, R, w] parameters
    - config: Configuration dictionary
    - disorder_std: Standard deviation for disorder visualization
    - save_path: Path to save the plot (optional)
    
    Returns:
    - matplotlib figure object
    """
    a, b, r, R, w = design_vector
    
    # Generate hole positions
    holes, num_cells = _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Full ring view
    ax1.set_aspect('equal')
    
    # Draw ring waveguide
    ring_outer = plt.Circle((0, 0), R + w/2, fill=False, color='blue', linewidth=2, label='Ring outer')
    ring_inner = plt.Circle((0, 0), R - w/2, fill=False, color='blue', linewidth=2, label='Ring inner')
    ax1.add_patch(ring_outer)
    ax1.add_patch(ring_inner)
    
    # Draw holes
    for i, (x, y, hole_r) in enumerate(holes):
        color = 'red' if i % 2 == 0 else 'orange'  # Alternate colors for a/b spacing
        hole = plt.Circle((x, y), hole_r, fill=True, color=color, alpha=0.7)
        ax1.add_patch(hole)
    
    # Set limits and labels
    margin = R + w/2 + 1
    ax1.set_xlim(-margin, margin)
    ax1.set_ylim(-margin, margin)
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'SSH Ring Resonator\nR={R:.2f}μm, w={w:.2f}μm, {len(holes)} holes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Zoomed view of unit cells
    ax2.set_aspect('equal')
    
    # Show first few unit cells in detail
    angle_per_hole = 2 * np.pi / len(holes)
    zoom_angles = np.arange(0, 6 * angle_per_hole, angle_per_hole)  # Show 6 holes
    
    # Draw arc section
    angles = np.linspace(zoom_angles[0] - angle_per_hole/2, 
                        zoom_angles[-1] + angle_per_hole/2, 100)
    x_outer = (R + w/2) * np.cos(angles)
    y_outer = (R + w/2) * np.sin(angles)
    x_inner = (R - w/2) * np.cos(angles)
    y_inner = (R - w/2) * np.sin(angles)
    
    ax2.plot(x_outer, y_outer, 'b-', linewidth=2, label='Waveguide boundary')
    ax2.plot(x_inner, y_inner, 'b-', linewidth=2)
    
    # Draw holes in zoom region
    for i, (x, y, hole_r) in enumerate(holes[:6]):
        color = 'red' if i % 2 == 0 else 'orange'
        hole = plt.Circle((x, y), hole_r, fill=True, color=color, alpha=0.7)
        ax2.add_patch(hole)
        
        # Add spacing annotations
        if i < 5:
            next_x, next_y = holes[i+1][:2]
            spacing = 'a' if i % 2 == 0 else 'b'
            mid_x, mid_y = (x + next_x)/2, (y + next_y)/2
            ax2.annotate(spacing, (mid_x, mid_y), fontsize=12, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set zoom limits
    zoom_margin = 1.5
    zoom_x_min = min(x for x, y, r in holes[:6]) - zoom_margin
    zoom_x_max = max(x for x, y, r in holes[:6]) + zoom_margin
    zoom_y_min = min(y for x, y, r in holes[:6]) - zoom_margin
    zoom_y_max = max(y for x, y, r in holes[:6]) + zoom_margin
    
    ax2.set_xlim(zoom_x_min, zoom_x_max)
    ax2.set_ylim(zoom_y_min, zoom_y_max)
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title(f'Unit Cell Detail\na={a:.3f}μm, b={b:.3f}μm, r={r:.3f}μm')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_geometry_properties(design_vector):
    """
    Analyze geometric properties of the design.
    
    Returns:
    - Dictionary with geometric analysis
    """
    a, b, r, R, w = design_vector
    
    # Basic parameters
    dimerization_ratio = a / b if b > 0 else float('inf')
    unit_cell_length = a + b
    circumference = 2 * np.pi * R
    num_unit_cells = circumference / unit_cell_length
    total_holes = int(num_unit_cells) * 2
    
    # Filling factors
    hole_area = np.pi * r**2
    unit_cell_area = w * unit_cell_length  # Approximate
    filling_factor = hole_area / unit_cell_area if unit_cell_area > 0 else 0
    
    # Band gap estimate (rough approximation)
    # For photonic crystals, larger dimerization typically opens larger gaps
    estimated_gap_width = dimerization_ratio * 0.1  # Rough estimate
    
    # Fabrication metrics
    min_feature = min(r, min(a, b), w)
    aspect_ratio = w / (2 * r) if r > 0 else float('inf')
    
    analysis = {
        'dimerization_ratio': dimerization_ratio,
        'unit_cell_length': unit_cell_length,
        'num_unit_cells': num_unit_cells,
        'total_holes': total_holes,
        'filling_factor': filling_factor,
        'estimated_gap_width': estimated_gap_width,
        'min_feature_size': min_feature,
        'aspect_ratio': aspect_ratio,
        'ring_circumference': circumference,
        'waveguide_area': np.pi * ((R + w/2)**2 - (R - w/2)**2),
    }
    
    return analysis

def validate_geometry_constraints(design_vector, config):
    """
    Validate that geometry meets fabrication and physical constraints.
    
    Returns:
    - List of constraint violations
    """
    a, b, r, R, w = design_vector
    violations = []
    
    # Get fabrication constraints if available
    fab_constraints = config.get('fabrication', {})
    min_feature = fab_constraints.get('min_feature_size', 0.05)
    max_aspect_ratio = fab_constraints.get('max_aspect_ratio', 10.0)
    
    # Minimum feature size
    if r < min_feature:
        violations.append(f"Hole radius {r:.3f} < minimum feature size {min_feature}")
    if a < min_feature:
        violations.append(f"Spacing 'a' {a:.3f} < minimum feature size {min_feature}")
    if b < min_feature:
        violations.append(f"Spacing 'b' {b:.3f} < minimum feature size {min_feature}")
    if w < min_feature:
        violations.append(f"Waveguide width {w:.3f} < minimum feature size {min_feature}")
    
    # Physical constraints
    if 2 * r >= w:
        violations.append(f"Hole diameter {2*r:.3f} >= waveguide width {w:.3f}")
    
    if a <= b:
        violations.append(f"No dimerization: a ({a:.3f}) <= b ({b:.3f})")
    
    # Aspect ratio
    aspect_ratio = w / (2 * r) if r > 0 else float('inf')
    if aspect_ratio > max_aspect_ratio:
        violations.append(f"Aspect ratio {aspect_ratio:.1f} > maximum {max_aspect_ratio}")
    
    # Ring geometry
    if R < w:
        violations.append(f"Ring radius {R:.3f} < waveguide width {w:.3f}")
    
    return violations

def create_geometry_report(design_vector, config, save_path=None):
    """
    Create a comprehensive geometry report with visualization and analysis.
    """
    analysis = analyze_geometry_properties(design_vector)
    violations = validate_geometry_constraints(design_vector, config)
    
    # Create visualization
    fig = visualize_ring_geometry(design_vector, config)
    
    # Create text report
    a, b, r, R, w = design_vector
    
    report = []
    report.append("# Geometry Analysis Report")
    report.append(f"Generated: {datetime.now()}")
    report.append("")
    
    report.append("## Design Parameters")
    report.append(f"- a (dimerization 1): {a:.4f} μm")
    report.append(f"- b (dimerization 2): {b:.4f} μm") 
    report.append(f"- r (hole radius): {r:.4f} μm")
    report.append(f"- R (ring radius): {R:.4f} μm")
    report.append(f"- w (waveguide width): {w:.4f} μm")
    report.append("")
    
    report.append("## Geometric Analysis")
    for key, value in analysis.items():
        if isinstance(value, float):
            report.append(f"- {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            report.append(f"- {key.replace('_', ' ').title()}: {value}")
    report.append("")
    
    if violations:
        report.append("## Constraint Violations")
        for violation in violations:
            report.append(f"- {violation}")
    else:
        report.append("## Constraint Validation")
        report.append("✓ All constraints satisfied")
    report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        # Save report
        report_path = save_path.replace('.png', '_report.md')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save visualization
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return report_text, analysis, violations