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
    length_in = length_cm * cm_to_in
    width_in = width_cm * cm_to_in
    square_size_in = square_size_cm * cm_to_in
    
    num_cols = int(length_cm / square_size_cm)   # horizontal, numbered axis
    num_rows = int(width_cm / square_size_cm)    # vertical, lettered axis
    
    # Format the key text (you can tweak this string if you want different units)
    key_text = f"Grid: {num_cols}x{num_rows}\n{length_cm}x{width_cm} cm\nCell: {square_size_cm*10:.0f} mm"
    
    letters = [chr(65 + i) for i in range(num_rows)]
    
    fig = plt.figure(figsize=(length_in, width_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, length_in)
    ax.set_ylim(0, width_in)
    ax.axis('off')

    for row in range(num_rows):
        letter = letters[row]
        y = row * square_size_in

        for col in range(num_cols):
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

    output_file = rf'{output_prefix}_{length_cm}x{width_cm}cm.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Checkerboard saved as {output_file}")
    print(f"   Physical size: {length_cm} cm × {width_cm} cm")
    print(f"   Grid: {num_cols} × {num_rows} cells, each {square_size_cm} cm ({square_size_cm*10:.0f} mm)")

# =============================================================================
# Example calls — change these to whatever you need
# =============================================================================
if __name__ == "__main__":
    # Example 1: 100 cm × 60 cm board → 50 × 30 grid of 2cm squares
    generate_checkerboard_pdf(length_cm=100, width_cm=60, square_size_cm=4, output_prefix='checkerboard')
    
    # Example 2: 80 cm × 40 cm board → 40 × 20 grid
    # generate_checkerboard_pdf(length_cm=80, width_cm=40, output_file='checkerboard_80x40cm.pdf')