import streamlit as st
from gec_metrics import get_metric_ids, get_metric, get_meta_eval_ids, get_meta_eval
from gec_metrics.metrics import (
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForSourceFree,
    inputs_handler,
)
from dataclasses import asdict
import pandas as pd
from pathlib import Path
import subprocess

def inputs_field(label='sources'):
    st.write(f"Enter {label}")
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area(f"Enter {label} (one per line)")
    with col2:
        uploaded_file = st.file_uploader(f"Or, upload {label} file", type=["txt"], key=label)
    
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    return text.rstrip().split('\n')

def main():
    if not Path('meta_eval_data').exists():
        subprocess.run([
            'gecmetrics-prepare-meta-eval'
        ])
    st.title("gec-metrics App")

    st.info("Since GPUs are unavailable in this demo, neural-based metrics such as IMPARA require significant computation time.")
    st.info("The SOME metric requires model downloading, so it is not supported in this demo.")
    st.info("ERRANT cannot be used because it appears the en_core_web_sm model cannot be downloaded on Streamlit Cloud.")
    st.info("The llmkobayashi24** metrics are not designed for evaluating a single system and therefore only works for meta-evaluation.")
    
    st.write('Choose a metric:')
    metric_id = st.selectbox("", get_metric_ids(), index=get_metric_ids().index('green'))
    if metric_id == 'errant':
        import spacy
        try:
            spacy.load('en_core_web_sm')
        except:
            subprocess.run("uv run python -m ensurepip --upgrade".split(' '))
            subprocess.run("uv run -m spacy download en_core_web_sm".split(' '))

    metric_class = get_metric(metric_id)        
    
    st.write("Choose the configurations:")
    config_values = {}
    if hasattr(metric_class, "Config"):
        config_fields = vars(metric_class.Config)
        for field, default in config_fields.items():
            if not field.startswith("_"):
                if isinstance(default, (int, float)):
                    config_values[field] = st.number_input(f"{field}", value=default, step=1 if isinstance(default, int) else 0.01)
                else:
                    config_values[field] = st.text_input(f"{field}", value=str(default))
    if not issubclass(metric_class, MetricBaseForSourceFree):
        sources = inputs_field('sources')
    hypotheses = inputs_field('hypotheses')
    
    if st.session_state.get('num_refs') is None:
        st.session_state['num_refs'] = 1
    references_list = []
    metric = None
    if not issubclass(metric_class, MetricBaseForReferenceFree):
        for ref_id in range(st.session_state['num_refs']):
            references = inputs_field(f'references{ref_id}')
            references_list.append(references)
        if st.button("Add references"):
            st.session_state['num_refs'] += 1

    if st.button("Evaluate"):
        try:
            config = metric_class.Config(**config_values)
            metric = metric_class(config)
            result = metric.score_corpus(
                **inputs_handler(metric, sources, hypotheses, references_list)
            )
            st.write("### Evaluation Result")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

    st.header("Meta-evaluation")
    meta_id = st.selectbox("", get_meta_eval_ids(), index=get_meta_eval_ids().index('seeda'))
    meta_class = get_meta_eval(meta_id)
    
    st.write("Choose the configurations:")
    config_values_meta = {}
    if hasattr(meta_class, "Config"):
        config_fields = vars(meta_class.Config)
        for field, default in config_fields.items():
            if not field.startswith("_"):
                if isinstance(default, (int, float)):
                    config_values_meta[field] = st.number_input(f"{field}", value=default, step=1 if isinstance(default, int) else 0.01)
                else:
                    config_values_meta[field] = st.text_input(f"{field}", value=str(default))
    do_window = st.checkbox("Do window analysis")
    if meta_id == 'seeda':
        sys_human_scores = ['ts_sent', 'ts_edit', 'ew_sent', 'ew_edit']
        sent_human_scores = ['sent', 'edit']
    elif meta_id == 'gjg':
        sys_human_scores = ['ts', 'ew']
        sent_human_scores = ['corr']
    if do_window:
        window_config = {
            'window': st.number_input(f"window:", value=4, step=1),
            'aggregation': st.selectbox('Aggregation method:',['default', 'trueskill']),
            'human_score': st.selectbox('Huamn score:', sys_human_scores)
        }
    do_pairwise = st.checkbox("Do pairwise analysis")
    if do_pairwise:
        pair_config = {
            'human_score': st.selectbox('Huamn score:', sent_human_scores)
        }
    if st.button("Meta-evaluate"):
        try:
            if metric is None:
                config = metric_class.Config(**config_values)
                metric = metric_class(config)
            config = meta_class.Config(**config_values_meta)
            meta = meta_class(config)
            result = meta.corr_system(
                metric
            )
            st.write("### Meta-evaluation Results")
            results_dict = asdict(result)
            data = {
                key: [f"{results_dict[key]['pearson']:.3f}", f"{results_dict[key]['spearman']:.3f}"] \
                for key in sorted(list(results_dict.keys()))
            }
            df = pd.DataFrame(data, index=['Pearson', 'Spearman'])
            st.table(df)

            if do_window:
                st.write('Window-analysis results.')
                human_score = window_config.pop('human_score')
                window_results = meta.window_analysis_system(metric, **window_config)
                window_fig = meta.window_analysis_plot(window_results.__dict__[human_score])
                st.pyplot(window_fig)
            if do_pairwise:
                st.write('Pairwise-analysis results.')
                pairwise_results = meta.pairwise_analysis(metric)
                pairwise_fig = meta.pairwise_analysis_plot(pairwise_results[pair_config['human_score']])
                st.pyplot(pairwise_fig)
        except Exception as e:
            st.error(f"Error: {e}")
    

if __name__ == "__main__":
    main()