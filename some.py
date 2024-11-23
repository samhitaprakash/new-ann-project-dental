# import matplotlib.pyplot as plt

# # Create a figure for the flowchart
# fig, ax = plt.subplots(figsize=(10, 12))
# ax.axis("off")

# # Box properties
# box_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="#e0f7fa")

# # Arrow properties
# arrow_props = dict(arrowstyle="->", color="black", lw=1.5)

# # Add boxes for the layers in the neural network
# layers = [
#     ("Input Layer\n(7 Features: MRB, BiCW, etc.)", (0.5, 0.9)),
#     ("Hidden Layer 1\n(50 Neurons, Sigmoid Activation)", (0.5, 0.75)),
#     ("Hidden Layer 2\n(100 Neurons, Sigmoid Activation)", (0.5, 0.6)),
#     ("Output Layer\n(2 Neurons: Male/Female, Linear Activation)", (0.5, 0.45))
# ]

# # Draw the boxes
# for text, pos in layers:
#     ax.text(pos[0], pos[1], text, ha="center", va="center", bbox=box_props, fontsize=10)

# # Add arrows connecting the layers
# connections = [
#     ((0.5, 0.87), (0.5, 0.78)),  # Input -> Hidden Layer 1
#     ((0.5, 0.72), (0.5, 0.63)),  # Hidden Layer 1 -> Hidden Layer 2
#     ((0.5, 0.57), (0.5, 0.48))   # Hidden Layer 2 -> Output Layer
# ]

# for start, end in connections:
#     ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props)

# # Display the flowchart
# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt

# # Create a figure for the flowchart
# fig, ax = plt.subplots(figsize=(10, 12))
# ax.axis("off")

# # Box properties
# box_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="#e0f7fa")

# # Arrow properties
# arrow_props = dict(arrowstyle="->", color="black", lw=1.5)

# # Add boxes for the layers in the neural network
# layers = [
#     ("Input Layer\n(7 Features: MRB, BiCW, etc.)", (0.5, 0.9)),
#     ("Hidden Layer 1\n(50 Neurons, Sigmoid Activation)", (0.5, 0.75)),
#     ("Hidden Layer 2\n(100 Neurons, Sigmoid Activation)", (0.5, 0.6)),
#     ("Output Layer\n(2 Neurons: Male/Female, Linear Activation)", (0.5, 0.45))
# ]

# # Draw the boxes
# for text, pos in layers:
#     ax.text(pos[0], pos[1], text, ha="center", va="center", bbox=box_props, fontsize=10)

# # Add arrows connecting the layers
# connections = [
#     ((0.5, 0.87), (0.5, 0.78)),  # Input -> Hidden Layer 1
#     ((0.5, 0.72), (0.5, 0.63)),  # Hidden Layer 1 -> Hidden Layer 2
#     ((0.5, 0.57), (0.5, 0.48))   # Hidden Layer 2 -> Output Layer
# ]

# for start, end in connections:
#     ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props)

# # Display the flowchart
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

# Create a figure for the network architecture
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

# Box properties for layers
box_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="#d9f2f7")

# Text for each layer
layers = [
    ("Input Layer\n(7 Features: MRB, BiCW, etc.)", (0.5, 0.9)),
    ("Hidden Layer 1\n(64 Neurons, ReLU Activation)", (0.5, 0.7)),
    ("Hidden Layer 2\n(32 Neurons, ReLU Activation)", (0.5, 0.5)),
    ("Output Layer\n(2 Neurons: Male/Female, Softmax Activation)", (0.5, 0.3))
]

# Add layer boxes
for text, pos in layers:
    ax.text(pos[0], pos[1], text, ha="center", va="center", bbox=box_props, fontsize=10)

# Add arrows between layers
arrows = [
    ((0.5, 0.87), (0.5, 0.73)),  # Input -> Hidden Layer 1
    ((0.5, 0.67), (0.5, 0.53)),  # Hidden Layer 1 -> Hidden Layer 2
    ((0.5, 0.47), (0.5, 0.33))   # Hidden Layer 2 -> Output Layer
]

arrow_props = dict(arrowstyle="->", color="black", lw=1.5)

for start, end in arrows:
    ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props)

# Display the diagram
plt.tight_layout()
plt.show()
