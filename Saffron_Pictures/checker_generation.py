import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def generate_checkerboard_pdf(length_cm, width_cm, square_size_cm=2.0, font_size_id=12, font_size_key=8, output_prefix='checkerboard'):
    """
    Generates a checkerboard PDF with:
    - 2cm squares (customizable)
    - Alphanumeric IDs on white squares (A0 bottom-left)
    - A clear key/legend INSIDE cell A0 specifying grid size and cell length
    
    Practical use: Print this once, and anyone (or your calibration script) can immediately
    know the physical scale without separate documentation.
    """
    cm_to_in = 1 / 2.54
    square_size_in = square_size_cm * cm_to_in
    
    # Snap to largest grid fitting within inputs (floor division)
    num_horizontal = int(length_cm / square_size_cm)  # Columns (numbers, horizontal axis)
    num_vertical = int(width_cm / square_size_cm)     # Rows (letters, vertical axis)
    
    # Compute snapped physical sizes
    snapped_horizontal_cm = num_horizontal * square_size_cm
    snapped_vertical_cm = num_vertical * square_size_cm
    horizontal_in = snapped_horizontal_cm * cm_to_in
    vertical_in = snapped_vertical_cm * cm_to_in
    
    # Format the key text: Grid horizontal × vertical, dimensions horizontal × vertical
    key_text = f"Grid: {num_horizontal}×{num_vertical}\n{snapped_horizontal_cm}×{snapped_vertical_cm} cm\nCell: {square_size_cm*10:.0f} mm"
    
    letters = [chr(65 + i) for i in range(num_vertical)]
    
    # Figure size: (horizontal width, vertical height) in inches
    fig = plt.figure(figsize=(horizontal_in, vertical_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, horizontal_in)
    ax.set_ylim(0, vertical_in)
    ax.axis('off')

    for row in range(num_vertical):
        letter = letters[row]
        y = row * square_size_in

        for col in range(num_horizontal):
            x = col * square_size_in
            is_white = (row + col) % 2 == 0
            color = 'white' if is_white else 'black'
            
            # Draw the square
            rect = Rectangle((x, y), square_size_in, square_size_in,
                             facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            if is_white:
                # Normal cell ID (e.g., A0, A1, B5, etc.)
                cell_id = f"{letter}{col}"
                ax.text(x + square_size_in / 2, y + square_size_in / 2,
                        cell_id, ha='center', va='center',
                        fontsize=font_size_id, color='black', weight='normal')
                
                # Legend/key ONLY in A0 (bottom-left cell)
                if row == 0 and col == 0:
                    ax.text(x + square_size_in / 2, y + square_size_in * 0.75,   # a bit higher
                            key_text, ha='center', va='center',
                            fontsize=font_size_key, color='black', weight='bold',
                            linespacing=0.9)

    # Use snapped sizes in filename (avoid floats with .0f)
    output_file = f'{output_prefix}_{snapped_horizontal_cm:.0f}x{snapped_vertical_cm:.0f}cm.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Checkerboard saved as {output_file}")
    print(f"   Physical size: {snapped_horizontal_cm} cm × {snapped_vertical_cm} cm")
    print(f"   Grid: {num_horizontal} × {num_vertical} cells, each {square_size_cm} cm ({square_size_cm*10:.0f} mm)")

# =============================================================================
# Example calls — change these to whatever you need
# =============================================================================
if __name__ == "__main__":
    # Example 1: Snaps to 88 × 104 cm board → 22 × 26 grid of 4cm squares
    generate_checkerboard_pdf(length_cm=104, width_cm=92, square_size_cm=4, output_prefix='checkerboard')
    
    # Example 2: 80 cm × 40 cm board → 40 × 20 grid
    # generate_checkerboard_pdf(length_cm=80, width_cm=40, square_size_cm=2, output_prefix='checkerboard')