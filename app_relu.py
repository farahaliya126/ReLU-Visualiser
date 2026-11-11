"""BSD3513 Lab 5 - App 1: ReLU activation function visualiser (Streamlit)."""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="ReLU Visualiser", page_icon="📈", layout="wide")
st.title("📈 ReLU (Rectified Linear Unit)")
st.latex(r"f(x)=\max(0,x)\qquad f'(x)=\begin{cases}1 & x>0\\0 & x\le 0\end{cases}")


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


# ---- Sidebar controls ----
st.sidebar.header("Controls")
x_min, x_max = st.sidebar.slider("x-axis range", -10.0, 10.0, (-6.0, 6.0), 0.5)
x_in = st.sidebar.slider("Input value x", float(x_min), float(x_max), 2.0, 0.1)
show_grad = st.sidebar.checkbox("Show derivative", True)

x = np.linspace(x_min, x_max, 400)
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, relu(x), label="ReLU(x)", color="tab:blue", lw=2)
    if show_grad:
        ax.plot(x, relu_grad(x), "--", label="Derivative", color="tab:orange")
    ax.scatter([x_in], [relu(np.array(x_in))], color="red", zorder=5, label="Your input")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.metric("ReLU(x)", f"{float(relu(np.array(x_in))):.4f}")
    st.metric("Derivative", f"{float(relu_grad(np.array(x_in))):.0f}")
    st.markdown(
        "**Key points**\n"
        "- Output range: [0, ∞)\n"
        "- Cheap to compute, reduces vanishing gradients\n"
        "- Weakness: 'dying ReLU' (neurons stuck at 0 when x ≤ 0)\n"
        "- Common in hidden layers of CNNs and deep nets"
    )
