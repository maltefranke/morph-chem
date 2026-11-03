"""Morph's public Gradio interface, with ZeroGPU generation."""

import html
import json
from pathlib import Path

import spaces  # Initialize ZeroGPU before importing torch through backend.

# isort: split
import gradio as gr
import pandas as pd
import py3Dmol
from trajectory_view import trajectory_viewer

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "models.json").read_text())


def description(key):
    spec = REGISTRY[key]
    folder = spec["checkpoint"].rsplit("/", 1)[0]
    url = f"https://huggingface.co/maltefranke/morph/tree/main/{folder}"
    return f"{spec['description']} [Model details]({url})."


def change_model(key):
    return (
        gr.update(visible=REGISTRY[key]["kind"] == "logp"),
        gr.update(visible=REGISTRY[key]["kind"] == "count"),
        description(key),
    )


def viewer(records, sample):
    item = next((r for r in records if r["index"] == int(sample or 0)), None)
    if item is None:
        return (
            "<p style='color:#65758b;padding:24px'>"
            "Generate molecules to explore their 3D structures.</p>"
        )
    view = py3Dmol.view(width="100%", height=410)
    if item.get("molblock"):
        view.addModel(item["molblock"], "sdf")
    else:
        view.addModel()
        view.getModel().addAtoms(item["frames"][-1]["atoms"])
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.23}})
    view.setBackgroundColor("#f7f9fc")
    view.zoomTo()
    document = view._make_html()
    return (
        '<iframe title="Interactive molecule" '
        'sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:425px;border:0;border-radius:12px" srcdoc="'
        + html.escape(document, quote=True)
        + '"></iframe>'
    )


@spaces.GPU(duration=45)
def sample_on_gpu(key, target_logp, target_n, count, seed):
    # Some scientific dependencies probe CUDA while importing. Import them
    # only after ZeroGPU attaches the real device, keeping its parent clean.
    from backend import MODELS, generate_graphs, load_models, present  # noqa: PLC0415

    if key not in MODELS:
        load_models("cuda", keys=[key])
    result = generate_graphs(
        key, target_logp, target_n, count, seed, trajectory=True, require_gpu=True
    )
    return present(
        result["graphs"],
        key,
        target_logp,
        target_n,
        seed,
        trajectories=result["trajectories"],
        times=result["times"],
        runtime=result["runtime"],
    )


def generate(key, target_logp, target_n, count, seed):
    gallery, rows, records, sdf, csv, status, archive, runtime = sample_on_gpu(
        key, target_logp, target_n, count, seed
    )
    choices = [str(record["index"]) for record in records]
    selected = choices[0] if choices else None
    table = pd.DataFrame(rows)[
        ["sample", "valid", "atoms", "logP", "target_logP", "target_N", "smiles"]
    ]
    return (
        gallery,
        table,
        records,
        gr.update(choices=choices, value=selected),
        viewer(records, selected),
        sdf,
        csv,
        status,
        gr.update(choices=choices, value=selected),
        trajectory_viewer(records, selected),
        archive,
        runtime,
    )


CSS = """
.gradio-container {max-width: 1250px !important; margin: auto;}
#hero {padding: 16px 4px 22px;}
#hero h1 {font-size: 40px; letter-spacing: -1.5px; margin-bottom: 6px;}
#hero p {font-size: 17px; color: #62738b;}
footer {display: none !important;}
"""

with gr.Blocks(
    title="Morph · Molecular playground",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
    delete_cache=(3600, 3600),
) as demo:
    gr.Markdown("# Morph\nExplore molecular generation in 3D.", elem_id="hero")
    records = gr.State([])
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            model = gr.Dropdown(
                [
                    (REGISTRY[key]["label"], key)
                    for key in ["geom", *[k for k in REGISTRY if k != "geom"]]
                ],
                value="geom",
                label="Model",
                interactive=True,
            )
            note = gr.Markdown(description("geom"))
            target_logp = gr.Slider(
                -8, 9, value=2, step=0.1, label="Target logP", visible=False
            )
            target_n = gr.Slider(
                3,
                120,
                value=40,
                step=1,
                label="Target atoms, including H",
                visible=False,
            )
            count = gr.Slider(
                1,
                8,
                value=4,
                step=1,
                label="Molecules",
            )
            seed = gr.Number(
                value=42,
                minimum=0,
                maximum=2147483647,
                precision=0,
                label="Random seed",
            )
            button = gr.Button("Generate molecules", variant="primary", size="lg")
            gr.Markdown(
                "Shared GPU · You may need to sign in to Hugging Face and wait in "
                "the queue. Changing the seed gives new samples."
            )
            gr.Markdown(
                "[Checkpoints & evaluation](https://huggingface.co/maltefranke/morph) · [Source](https://huggingface.co/spaces/maltefranke/morph-playground/tree/main)"
            )
        with gr.Column(scale=2, min_width=420):
            status = gr.Markdown("Choose a model and generate your first molecules.")
            with gr.Tabs():
                with gr.Tab("Molecules"):
                    gallery = gr.Gallery(
                        label="Generated molecules",
                        columns=2,
                        height=600,
                        object_fit="contain",
                        show_label=False,
                    )
                with gr.Tab("3D view"):
                    selected = gr.Dropdown(
                        choices=[], label="Molecule", interactive=True
                    )
                    structure = gr.HTML(viewer([], None))
                    gr.Markdown("Drag to rotate · Scroll to zoom")
                with gr.Tab("Trajectory"):
                    trajectory_selected = gr.Dropdown(
                        choices=[], label="Molecule", interactive=True
                    )
                    trajectory_player = gr.HTML(trajectory_viewer([], None))
                    trajectory_file = gr.File(
                        label="Download trajectories (JSON + XYZ)"
                    )
                with gr.Tab("Properties"):
                    table = gr.Dataframe(
                        interactive=False, wrap=True, label="Every generation attempt"
                    )
            with gr.Row():
                sdf = gr.File(label="Download 3D structures (SDF)")
                csv = gr.File(label="Download properties (CSV)")
    gr.Markdown(
        "Generated targets are approximate. Chemical validity checks sanitization, "
        "connectedness and radicals (plus neutrality for QM9); it does not assess "
        "3D geometry. Samples are shown without filtering or retrying. N includes "
        "hydrogens except for the ChEMBL3D heavy-atom model."
    )
    model.change(change_model, model, [target_logp, target_n, note], queue=False)
    selected.change(viewer, [records, selected], structure, queue=False)
    trajectory_selected.change(
        trajectory_viewer,
        [records, trajectory_selected],
        trajectory_player,
        queue=False,
    )
    runtime_info = gr.JSON(visible=False)
    button.click(
        generate,
        [model, target_logp, target_n, count, seed],
        [
            gallery,
            table,
            records,
            selected,
            structure,
            sdf,
            csv,
            status,
            trajectory_selected,
            trajectory_player,
            trajectory_file,
            runtime_info,
        ],
        api_name="generate",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    # Download once at startup; model construction runs in the GPU worker.
    for spec in REGISTRY.values():
        hf_hub_download(
            "maltefranke/morph", spec["checkpoint"], revision=spec["revision"]
        )
    demo.queue(max_size=24).launch(ssr_mode=False)
