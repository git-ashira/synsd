"""
Streamlit app (GUI) for training and using SynSD models.

Author: Amandeep Singh Hira
email: ahira1@ualberta.ca
Date: September 2025

References: 

ualbertaIGEM. (2025). Ashbloom https://2025.igem.wiki/ualberta.

test.svg: https://www.svgrepo.com/svg/530662/ribosome

RNA emoji: https://emojidb.org/rna-emojis?utm_source=user_search

The pandas development team. (2020). pandas [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3509134

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Oliphant, T. E. (2020). NumPy [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.4147899

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch [Computer software]. https://pytorch.org/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). scikit-learn: Machine learning in Python [Computer software]. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org/

Streamlit Inc. (2023). Streamlit [Computer software]. https://streamlit.io/

Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., ... de Hoon, M. J. L. (2009). Biopython [Computer software]. Bioinformatics, 25(11), 1422–1423. https://biopython.org/

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment [Computer software]. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. (2021). Seaborn [Computer software]. https://seaborn.pydata.org/

Lorenz, R., Bernhart, S. H., Höner zu Siederdissen, C., Tafer, H., Flamm, C., Stadler, P. F., & Hofacker, I. L. (2011). ViennaRNA Package [Computer software]. https://www.tbi.univie.ac.at/RNA/

BioRender. (n.d.). BioRender [Computer software]. https://biorender.com/


Description: 
A ribosome binding site (RBS) refers to a brief segment of mRNA that plays a crucial role in attracting the ribosome to start the translation process, thus influencing the efficiency of protein production. For microbial synthetic biology, where meticulous regulation of gene expression is essential, the precise prediction and design of RBS sequences are vital.
Current RBS prediction models predominantly rely on RNA thermodynamics. The most prevalent method is created by the Salis Lab called Denovo DNA [1]. The Denovo DNA estimates RBS sequences by assessing the minimum folding energy of the mRNA alongside the ribosome binding energy, establishing a thermodynamic basis for RBS design [1].
In contrast, we developed a novel deep learning model to predict ribosome binding sites (RBS) and spacer sequences from mRNA contexts. Inspired by natural sequence-to-function mappings, we employ a Transformer encoder–decoder architecture, capable of learning long-range dependencies in RNA. The goal is to enable rational RBS+spacer design for microbial synthetic biology.
The model is coupled with the extraction and analysis code to form a pipeline for easier user interaction. The pipeline takes an annotated genome as an input and extracts out all the ribosome binding sequences and spacer sequences with their respective mRNA sequences. The resulting sequences are used to train the model for sequence prediction. 
The model is validated by calculating the minimum folding energies of the first 25 base pairs of the RNA and the ribosome-RNA sequence binding affinity. As the paper by Chen, Yi-Lan, and Jin-Der Wen elucidates mRNA–ribosome complexes that use less favorable / more structured RBS tend to be disfavored during initiation, via kinetic discrimination [2].

1. Reis, A.C. & Salis, H.M. (2020). An automated model test system for systematic development and improvement of gene expression models. ACS Synthetic Biology, 9(11), 3145-3156.
2. Chen, Yi-Lan, and Jin-Der Wen. "Translation initiation site of mRNA is selected through dynamic interaction with the ribosome." Proceedings of the National Academy of Sciences 119.22 (2022): e2118099119.



"""

import os
import io
import time
import random
import streamlit as st
from streamlit.components.v1 import html
import pipeline_func as pf
import threading
import synsd_transformer as trans
import torch

st.set_page_config(page_title="SynSD: Synthetic Shine-Dalgarno", page_icon="〰️", layout="wide")


# Session state helpers

if "logs" not in st.session_state:
    st.session_state.logs = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "model_id" not in st.session_state:
    st.session_state.model_id = ""
if "train_progress" not in st.session_state:
    st.session_state.train_progress = 0
if "pred_rbs" not in st.session_state:
    st.session_state.pred_rbs = ""
if "pred_spacer" not in st.session_state:
    st.session_state.pred_spacer = ""
if "training_file" not in st.session_state:
    st.session_state.training_file = ""
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "training_start_time" not in st.session_state:
    st.session_state.training_start_time = None
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

def log(msg: str):
    st.session_state.logs.append(msg)

def alert(msg: str):
    js_code = f"""<script> alert("{msg}");</script>"""
    html(js_code, height=0, width=0)



# Header

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("logo.png", width=400)
st.markdown("<h1 style='text-align: center;'>SynSD: Synthetic Shine-Dalgarno</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Train a model from your genome + annotations or load an existing model, then predict RBS + spacer for a given mRNA.</h3>", unsafe_allow_html=True)



# Tabs: Train new model | Use existing model

train_tab, existing_tab = st.tabs(["Train new model", "Use existing model"])  # mirrors the React tabs

with train_tab:
    st.subheader("Upload inputs")
    name_of_new_model = st.text_input("Name of your model", help="Give your model a name", value=f"rbs-model-name")
    col_a, col_b = st.columns(2)
    with col_a:
        genome_file = st.file_uploader("Genome.fasta", type=["fa", "fasta", "fna"], help="Provide the reference genome FASTA")
    with col_b:
        tsv_file = st.file_uploader("Annotation.tsv", type=["tsv"], help="Provide the annotation table")

    anti_sd = st.text_input("Anti-SD sequence (5'-3')", value="CCUCCUUA", help="The anti-Shine-Dalgarno sequence (default: CCUCCUUA for E. coli)")
    rbs_upstream = st.number_input("RBS upstream length", min_value=5, max_value=50, value=25, step=1, help="Number of bases upstream of start codon to search for RBS sequence")

    # data cleaning/preprocessing functions:
    def pre_process():
        # Create temp directory if it doesn't exist
        os.makedirs("temp_data_files", exist_ok=True)

        genome_path = f"temp_data_files/{genome_file.name}"
        tsv_path = f"temp_data_files/{tsv_file.name}"
        content_string_genome = genome_file.getvalue().decode('utf-8')
        content_string_tsv = tsv_file.getvalue().decode('utf-8')
        
        print(f"[preprocess] Saved genome to {genome_path} and annotation to {tsv_path} and type {type(content_string_genome)}")

        # Call the pipeline functions with file paths (not UploadedFile objects)
        anti_sd_alignment = pf.anti_sd_converter_for_alignment(anti_sd)
        rbs_sequence = pf.extract_gene_data(content_string_tsv, content_string_genome, rbs_upstream=rbs_upstream)  # Use file paths
        output_csv = f'temp_data_files/alignment_results_{name_of_new_model}.csv'
        pf.process_sequences_from_df(rbs_sequence, output_csv, anti_sd_alignment)

        training_file = output_csv
        st.session_state.training_file = training_file

        message = f"Pre-processing completed! Training data saved as alignment_results_{name_of_new_model}.csv"
        log(message)
        alert(message)

        return training_file


    st.button("Start pre-processing", type="primary", disabled=genome_file is None or tsv_file is None, on_click=pre_process)

    st.subheader("Training settings")
    col1, col2 = st.columns([1,1])
    with col1:
        epochs = st.number_input("Epochs", min_value=1, value=20, step=1)
    with col2:
        size_batch = st.number_input("Batch size", min_value=1, value=16, step=1)

    can_train = st.session_state.training_file != "" and epochs > 0 and size_batch > 0

    # training functions
    def start_training():
        st.session_state.is_training = True
        st.session_state.training_start_time = time.time()
        log(f"[train] Starting training with {st.session_state.training_file}, epochs={epochs}, batch_size={size_batch}")
        training_csv = st.session_state.training_file
        # --- TRAINING THREAD ---
        def training_thread():
            try:
                model_name = f"models/{name_of_new_model}.pth"
                trans.main_train(model_name=model_name, dataset_csv=training_csv, num_epochs=epochs, batch_size=size_batch)
                # log(f"[train] Training completed successfully!")
            except Exception as e:
                # log(f"[train] Error occurred: {e}")
                pass
            finally:
                st.session_state.is_training = False
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    start_train = st.button("Start training", type="primary", disabled=not can_train, on_click=start_training)


with existing_tab:
    st.subheader("Select a model")
    colx, coly = st.columns(2)
    file_list = pf.check_data_files(folder_path="models")
    options = [file_path for file_path in file_list if file_path.endswith(".pth")]

    with colx:
        existing_model = st.selectbox("Registry", options, index=0)
        if existing_model:
            st.success(f"Selected model: {existing_model}")
    with coly:
        upload_model = st.file_uploader("Upload checkpoint (.pth)", type=["pth"])  

    if st.button("Load model", disabled=(not existing_model and upload_model is None)):

        if upload_model is not None:
            os.makedirs("./models", exist_ok=True)
            ckpt_path = os.path.join("models", upload_model.name)
            with open(ckpt_path, "wb") as f:
                f.write(upload_model.read())
            st.session_state.model_id = upload_model.name
        else:
            st.session_state.model_id = existing_model

        st.session_state.model_ready = True
        log("[model] Existing model loaded and ready.")
        st.success("Model loaded and ready.")


# Prediction section

    st.markdown("---")
    st.subheader("RBS Prediction")
    folding_energy = 0
    binding_energy = 0
    structure = "temp_data_files/test.svg"
    predicted_full_rbs = "-"
    full_pred = "-"



    left, right = st.columns([2,1])
    with left:

        mrna = st.text_area("mRNA sequence", value= "", height=180, placeholder="Paste mRNA sequence (A/C/G/U/T)")
        demo, run_pred = st.columns([1,1])
        if demo.button("Load demo"):
            demo_value = "ATGAATCTGATGACGACGATAACAGGCGTTGTGCTGGCAGGCGGTAAAGCCAGACGAA"
            # by clicking demo, the text_area gets disabled
            st.session_state.demo_loaded = True
            

            mrna = demo_value
        if run_pred.button("Predict", type="primary", disabled=not st.session_state.model_ready or len(mrna.strip()) == 0):
            log("[predict] Running inference…")
            model = st.session_state.model_id
            mrna_sequence = mrna.upper().replace(" ", "").replace("\n", "")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                model, vocab, idx_to_token = trans.load_trained_model(model, device)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
            
            if model == "":
                alert("No model loaded!")
                log("[predict] No model loaded, cannot run prediction.")
            else:
                rbs_pred, spacer_pred, full_pred = trans.predict_rbs_spacer(model, mrna_sequence, vocab, idx_to_token, max_length=50, device=device)
                st.session_state.pred_rbs = rbs_pred
                st.session_state.pred_spacer = spacer_pred
                predicted_full_rbs = full_pred
                folding_energy, binding_energy = pf.total_energy(full_pred, mrna_sequence)
                if full_pred != "-":
                    structure = f"./temp_data_files/{full_pred}.svg"
                log("[predict] Done.")

    with right:
        if full_pred != "-":
            structure = f"./temp_data_files/{full_pred}.svg"
        st.write("**Predicted RBS+spacer**")
        st.code(st.session_state.pred_rbs or "—")
        st.write("**Predicted Folding Energy of first 50 bases (Kcal/mol)**")
        st.code(folding_energy)
        st.write("**Predicted Binding Energy (using ACCUCCUUA as Anti-SD) (Kcal/mol)**")
        st.code(binding_energy)
        st.write("**Predicted Folding structure of first 50 bases**")
        st.image(structure or "./temp_data_files/file.svg", caption="Predicted structure", width="stretch")


# Console / Logs

st.markdown("---")
st.subheader("Console")
st.caption("Logs & messages from training and inference.")

# reload logs every second if training
if st.session_state.is_training:
    time.sleep(1)

    log_file_path = 'training_log.txt'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        st.text_area("Training Logs", value=log_content, height=300, disabled=True)
    else:
        st.info("No training logs yet.")
    st.rerun()
else:
    if st.session_state.logs:
        st.text_area("Logs", value="\n".join(st.session_state.logs), height=300, disabled=True)
    else:
        st.info("No logs yet.")

# footer 
st.markdown("---")  
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>"
    "Developed by Amandeep Singh Hira | University of Alberta iGEM 2025 | "
    "<a href='https://2025.igem.wiki/ualberta'>Wiki</a> | "
    "<a href='https://www.linkedin.com/in/amandeep-singh-hira-5a0bb1191/'>LinkedIn</a>"
    "</p>", 
    unsafe_allow_html=True
)