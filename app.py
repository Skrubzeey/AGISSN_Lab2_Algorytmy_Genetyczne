import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from fpdf import FPDF

from genetic_algorithm_base import BaseAG

from continous_funcs_and_tsp import (
    sphere_function,
    rosenbrock_function,
    rastrigin_function,
    ackley_function,
    generate_tsp_data,
    create_tsp_fitness
)

from selection_types import (
    tournament_selection,
    roulette_selection,
    ranking_selection
)

from mutation import (
    gaussian_mutation,
    uniform_mutation,
    swap_mutation,
    inverse_mutation
)

from crossover import (
    one_point_crossover,
    two_point_crossover,
    arithmetic_crossover,
    order_crossover
)

from encoding import (
    RealEncoding,
    PermutationEncoding
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Genetic Algorithm Laboratory",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "results" not in st.session_state:
    st.session_state.results = None

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def display_plot(
        fig,
        width=4,
        height=2.5,
        dpi=140
):

    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)

    for ax in fig.axes:

        ax.title.set_fontsize(10)

        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)

        ax.tick_params(
            axis='both',
            labelsize=7
        )

        legend = ax.get_legend()

        if legend is not None:
            legend.prop.set_size(7)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )


# =====================================================
def plot_convergence(history):

    fig, ax = plt.subplots()

    generations = np.arange(len(history))

    ax.plot(
        generations,
        history,
        linewidth=2,
        label="Best Fitness"
    )

    ax.set_title("Fitness Over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    ax.grid(True)
    ax.legend()

    return fig


# =====================================================
def plot_histogram(scores):

    fig, ax = plt.subplots()

    ax.hist(scores, bins=10)

    ax.set_title("Histogram of Results")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Frequency")

    return fig


# =====================================================
def plot_boxplot(scores):

    fig, ax = plt.subplots()

    ax.boxplot(scores)

    ax.set_title("Boxplot of Results")

    return fig


# =====================================================
def plot_tsp(coords, route):

    fig, ax = plt.subplots()

    ax.scatter(coords[:, 0], coords[:, 1], c="red")

    for i in range(len(route) - 1):

        p1 = route[i]
        p2 = route[i + 1]

        ax.plot(
            [coords[p1, 0], coords[p2, 0]],
            [coords[p1, 1], coords[p2, 1]],
            "b-"
        )

    p1 = route[-1]
    p2 = route[0]

    ax.plot(
        [coords[p1, 0], coords[p2, 0]],
        [coords[p1, 1], coords[p2, 1]],
        "b-"
    )

    ax.set_title("Best TSP Route")

    return fig


# =====================================================
def plot_contour(func, bounds, best_individual=None):

    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)

    X, Y = np.meshgrid(x, y)

    Z = np.array([
        func(np.array([xi, yi]))
        for xi, yi in zip(X.flatten(), Y.flatten())
    ])

    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=50
    )

    plt.colorbar(contour)

    if best_individual is not None:

        ax.scatter(
            best_individual[0],
            best_individual[1],
            color="red",
            s=100,
            label="Best"
        )

        ax.legend()

    ax.set_title("Contour Plot")

    return fig


# =====================================================
# PDF EXPORT
# =====================================================
def create_pdf(
        problem_type,
        function_name,
        params,
        stats_df,
        convergence_plot,
        histogram_plot=None,
        boxplot_plot=None,
        contour_plot=None,
        tsp_plot=None
):

    pdf = FPDF()

    # =================================================
    # PAGE 1
    # =================================================
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)

    pdf.cell(
        200,
        10,
        "Genetic Algorithm Benchmark Report",
        0,
        1,
        'C'
    )

    pdf.ln(10)

    # =================================================
    # CONFIG
    # =================================================
    pdf.set_font("Arial", 'B', 12)

    pdf.cell(
        200,
        10,
        "1. Experiment Configuration",
        0,
        1
    )

    pdf.set_font("Arial", size=11)

    pdf.cell(
        200,
        10,
        f"Problem Type: {problem_type}",
        0,
        1
    )

    pdf.cell(
        200,
        10,
        f"Function: {function_name}",
        0,
        1
    )

    for k, v in params.items():

        pdf.cell(
            200,
            10,
            f"{k}: {v}",
            0,
            1
        )

    pdf.ln(5)

    # =================================================
    # STATS
    # =================================================
    pdf.set_font("Arial", 'B', 12)

    pdf.cell(
        200,
        10,
        "2. Statistical Results",
        0,
        1
    )

    pdf.set_font("Arial", size=10)

    for _, row in stats_df.iterrows():

        pdf.cell(
            200,
            10,
            f"{row['Metric']}: {row['Value']}",
            0,
            1
        )

    # =================================================
    # PAGE 2
    # =================================================
    pdf.add_page()

    plots = [

        ("Convergence Plot", "convergence.png", convergence_plot),

        ("Histogram", "histogram.png", histogram_plot),

        ("Boxplot", "boxplot.png", boxplot_plot),

        ("Contour Plot", "contour.png", contour_plot),

        ("TSP Route", "tsp.png", tsp_plot)
    ]

    for title, filename, fig in plots:

        if fig is not None:

            try:

                if pdf.get_y() > 180:
                    pdf.add_page()

                pdf.set_font("Arial", 'B', 11)

                pdf.cell(
                    200,
                    10,
                    title,
                    0,
                    1
                )

                fig.savefig(
                    filename,
                    bbox_inches='tight',
                    dpi=150
                )

                pdf.image(
                    filename,
                    x=15,
                    y=pdf.get_y(),
                    w=160
                )

                pdf.ln(90)

            except Exception:

                pdf.cell(
                    200,
                    10,
                    f"Cannot add {title}",
                    0,
                    1
                )

    out = pdf.output(dest='S')

    if isinstance(out, (bytes, bytearray)):
        return bytes(out)

    elif isinstance(out, str):
        return out.encode('latin-1')

    else:
        return str(out).encode('latin-1')


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙ Genetic Algorithm Laboratory")

problem_type = st.sidebar.selectbox(
    "Problem Type",
    ["Continuous", "TSP"]
)

FUNCTIONS = {

    "Sphere": {
        "func": sphere_function,
        "bounds": (-5, 5)
    },

    "Rosenbrock": {
        "func": rosenbrock_function,
        "bounds": (-5, 10)
    },

    "Rastrigin": {
        "func": rastrigin_function,
        "bounds": (-5.12, 5.12)
    },

    "Ackley": {
        "func": ackley_function,
        "bounds": (-32, 32)
    }
}

# =====================================================
# PROBLEM CONFIG
# =====================================================
if problem_type == "Continuous":

    function_name = st.sidebar.selectbox(
        "Objective Function",
        list(FUNCTIONS.keys())
    )

    dimensions = st.sidebar.selectbox(
        "Dimensions",
        [2, 5, 10]
    )

    objective_function = FUNCTIONS[function_name]["func"]

    bounds = FUNCTIONS[function_name]["bounds"]

    encoding = RealEncoding(
        dimension=dimensions,
        bounds=bounds
    )

else:

    function_name = "TSP"

    n_cities = st.sidebar.selectbox(
        "Number of Cities",
        [10, 20, 50]
    )

    coords, dist_matrix = generate_tsp_data(
        n_cities
    )

    objective_function = create_tsp_fitness(
        dist_matrix
    )

    encoding = PermutationEncoding(
        n=n_cities
    )

# =====================================================
# GA PARAMETERS
# =====================================================
st.sidebar.subheader("Genetic Algorithm Parameters")

population_size = st.sidebar.selectbox(
    "Population Size",
    [10, 20, 50, 100],
    index=2
)

num_generations = st.sidebar.slider(
    "Generations",
    10,
    500,
    100
)

pc = st.sidebar.selectbox(
    "Crossover Probability (Pc)",
    [0.6, 0.8, 1.0],
    index=1
)

pm = st.sidebar.selectbox(
    "Mutation Probability (Pm)",
    [0.01, 0.05, 0.1],
    index=1
)

patience = st.sidebar.slider(
    "Patience",
    5,
    100,
    20
)

n_runs = st.sidebar.slider(
    "Number of Runs",
    1,
    50,
    20
)

# =====================================================
# SELECTION
# =====================================================
selection_name = st.sidebar.selectbox(
    "Selection Method",
    [
        "Tournament",
        "Roulette",
        "Ranking"
    ]
)

tournament_size = 3

if selection_name == "Tournament":

    tournament_size = st.sidebar.slider(
        "Tournament Size",
        2,
        10,
        3
    )

# =====================================================
# CROSSOVER
# =====================================================
if problem_type == "Continuous":

    crossover_name = st.sidebar.selectbox(
        "Crossover Method",
        [
            "One Point",
            "Two Point",
            "Arithmetic"
        ]
    )

else:

    crossover_name = st.sidebar.selectbox(
        "Crossover Method",
        ["OX"]
    )

# =====================================================
# MUTATION
# =====================================================
if problem_type == "Continuous":

    mutation_name = st.sidebar.selectbox(
        "Mutation Method",
        [
            "Gaussian",
            "Uniform"
        ]
    )

else:

    mutation_name = st.sidebar.selectbox(
        "Mutation Method",
        [
            "Swap",
            "Inverse"
        ]
    )

# =====================================================
# STRATEGIES
# =====================================================
selection_map = {

    "Tournament": lambda pop, fit:
        tournament_selection(
            pop,
            fit,
            tournament_size=tournament_size
        ),

    "Roulette": roulette_selection,

    "Ranking": ranking_selection
}

if problem_type == "Continuous":

    crossover_map = {

        "One Point": one_point_crossover,

        "Two Point": two_point_crossover,

        "Arithmetic": arithmetic_crossover
    }

else:

    crossover_map = {
        "OX": order_crossover
    }

if problem_type == "Continuous":

    mutation_map = {

        "Gaussian": lambda ind:
            gaussian_mutation(
                ind,
                mutation_prob=pm,
                bounds=bounds
            ),

        "Uniform": lambda ind:
            uniform_mutation(
                ind,
                mutation_prob=pm,
                bounds=bounds
            )
    }

else:

    mutation_map = {

        "Swap": lambda ind:
            swap_mutation(
                ind,
                mutation_prob=pm
            ),

        "Inverse": lambda ind:
            inverse_mutation(
                ind,
                mutation_prob=pm
            )
    }

# =====================================================
# MAIN PAGE
# =====================================================
st.title("🧬 Genetic Algorithm Benchmark Laboratory")

st.markdown(
    "Interactive environment for benchmarking genetic algorithms."
)

# =====================================================
# CONFIGURATION
# =====================================================
st.subheader("📋 Current Configuration")

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:

    st.markdown("### 🧩 Problem")

    st.write(f"**Type:** {problem_type}")

    if problem_type == "Continuous":

        st.write(f"**Function:** {function_name}")
        st.write(f"**Dimensions:** {dimensions}")
        st.write(f"**Bounds:** {bounds}")

    else:

        st.write(f"**Cities:** {n_cities}")

with config_col2:

    st.markdown("### 🧬 Genetic Algorithm")

    st.write(f"**Population:** {population_size}")
    st.write(f"**Generations:** {num_generations}")
    st.write(f"**Pc:** {pc}")
    st.write(f"**Pm:** {pm}")
    st.write(f"**Patience:** {patience}")
    st.write(f"**Runs:** {n_runs}")

with config_col3:

    st.markdown("### ⚙ Operators")

    st.write(f"**Selection:** {selection_name}")

    if selection_name == "Tournament":
        st.write(f"**Tournament Size:** {tournament_size}")

    st.write(f"**Crossover:** {crossover_name}")
    st.write(f"**Mutation:** {mutation_name}")

# =====================================================
# RUN BENCHMARK
# =====================================================
if st.button(
    "🚀 RUN BENCHMARK",
    use_container_width=True
):

    st.info("Benchmark execution started...")

    all_scores = []

    best_global_individual = None
    best_global_fitness = np.inf

    final_history = []

    progress_bar = st.progress(0)

    status_text = st.empty()

    start_time = time.time()

    for run in range(n_runs):

        status_text.text(
            f"Running benchmark {run + 1}/{n_runs}"
        )

        ga = BaseAG(

            population_size=population_size,

            fitness_func=objective_function,

            crossover_prob=pc,
            mutation_prob=pm,

            selection_method=selection_map[
                selection_name
            ],

            crossover_method=crossover_map[
                crossover_name
            ],

            mutation_method=mutation_map[
                mutation_name
            ],

            encoding=encoding
        )

        best_individual, best_fitness, history = ga.run(
            generations=num_generations,
            patience=patience
        )

        all_scores.append(best_fitness)

        if best_fitness < best_global_fitness:

            best_global_fitness = best_fitness
            best_global_individual = best_individual
            final_history = history

        progress_bar.progress(
            (run + 1) / n_runs
        )

    total_time = time.time() - start_time

    stats_df = pd.DataFrame({

        "Metric": [
            "Mean",
            "Std",
            "Best",
            "Worst",
            "Execution Time (s)"
        ],

        "Value": [
            np.mean(all_scores),
            np.std(all_scores),
            np.min(all_scores),
            np.max(all_scores),
            round(total_time, 2)
        ]
    })

    convergence_fig = plot_convergence(
        final_history
    )

    histogram_fig = plot_histogram(
        all_scores
    )

    boxplot_fig = plot_boxplot(
        all_scores
    )

    contour_fig = None
    tsp_fig = None

    if problem_type == "Continuous":

        if dimensions == 2:

            contour_fig = plot_contour(
                objective_function,
                bounds,
                best_global_individual
            )

    else:

        tsp_fig = plot_tsp(
            coords,
            best_global_individual
        )

    st.session_state.results = {

        "stats": stats_df,

        "convergence_fig": convergence_fig,

        "histogram_fig": histogram_fig,

        "boxplot_fig": boxplot_fig,

        "contour_fig": contour_fig,

        "tsp_fig": tsp_fig,

        "best_fitness": best_global_fitness,

        "best_individual": best_global_individual,

        "scores": all_scores
    }

    status_text.text(
        "Benchmark completed!"
    )

    st.rerun()

# =====================================================
# RESULTS
# =====================================================
if st.session_state.results is not None:

    st.divider()

    st.header("📊 Benchmark Results")

    results = st.session_state.results

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("Convergence Plot")

        display_plot(
            results["convergence_fig"]
        )

    with col2:

        st.subheader("Statistics")

        st.table(
            results["stats"]
        )

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Histogram")

        display_plot(
            results["histogram_fig"]
        )

    with col4:

        st.subheader("Boxplot")

        display_plot(
            results["boxplot_fig"]
        )

    if results["contour_fig"] is not None:

        st.subheader("Contour Plot")

        display_plot(
            results["contour_fig"],
            width=6,
            height=4
        )

    if results["tsp_fig"] is not None:

        st.subheader("TSP Visualization")

        display_plot(
            results["tsp_fig"],
            width=6,
            height=4
        )

    st.divider()

    st.subheader("📥 Export Results")

    csv_data = results["stats"].to_csv(
        index=False
    ).encode("utf-8")

    col_export1, col_export2 = st.columns(2)

    with col_export1:

        st.download_button(

            label="📄 Download CSV",

            data=csv_data,

            file_name="ga_results.csv",

            mime="text/csv"
        )

    with col_export2:

        pdf_params = {

            "Population Size": population_size,
            "Generations": num_generations,
            "Pc": pc,
            "Pm": pm,
            "Patience": patience,
            "Runs": n_runs,
            "Selection": selection_name,
            "Crossover": crossover_name,
            "Mutation": mutation_name
        }

        pdf_bytes = create_pdf(

            problem_type=problem_type,

            function_name=function_name,

            params=pdf_params,

            stats_df=results["stats"],

            convergence_plot=results["convergence_fig"],

            histogram_plot=results["histogram_fig"],

            boxplot_plot=results["boxplot_fig"],

            contour_plot=results["contour_fig"],

            tsp_plot=results["tsp_fig"]
        )

        st.download_button(

            label="📄 Download PDF Report",

            data=pdf_bytes,

            file_name="ga_report.pdf",

            mime="application/pdf"
        )