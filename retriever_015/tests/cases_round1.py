from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metadata import Metadata

# ------------------------------------------------------------
# test_case_1
# ------------------------------------------------------------
test_case_1 = {
    "name": 'case_1_medical_dl',
    "query": 'What are the effects of deep learning on medical image diagnosis?',
    "k1": 4,
    "metadatas": [
        Metadata(
            title='Deep Learning for Medical Image Diagnosis',
            year='2023',
            source_type='paper',
            id='paper_001',
            text_summary='A review of deep learning methods for medical image diagnosis, including CNNs and transformers.',
            text='This paper reviews the application of deep learning to medical image diagnosis.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Deep Learning in Natural Language Processing',
            year='2022',
            source_type='paper',
            id='paper_002',
            text_summary='A survey of deep learning applications in NLP.',
            text='This paper discusses transformers and recurrent neural networks for language processing.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Medical Image Segmentation with Convolutional Networks',
            year='2021',
            source_type='paper',
            id='paper_003',
            text_summary='Convolutional neural networks for segmentation of medical images.',
            text='The study evaluates CNN based methods for medical image segmentation.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='A Survey of Computer Vision',
            year='2020',
            source_type='paper',
            id='paper_004',
            text_summary='General computer vision techniques and applications.',
            text='The paper provides an overview of computer vision algorithms.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 4,
     'first_id': 'paper_001',
     'contains_ids': ['paper_001', 'paper_003', 'paper_004'],
     'order_before': [('paper_001', 'paper_003'), ('paper_003', 'paper_004')]}
,
}

# ------------------------------------------------------------
# test_case_2
# ------------------------------------------------------------
test_case_2 = {
    "name": 'case_2_transformer_protein',
    "query": 'How does transformer architecture improve protein structure prediction?',
    "k1": 4,
    "metadatas": [
        Metadata(
            title='Transformers for Protein Structure Prediction',
            year='2024',
            source_type='paper',
            id='paper_101',
            text_summary='Transformer-based models for predicting protein structures from amino acid sequences.',
            text='We investigate transformer architectures for protein structure prediction.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Protein Structure Prediction with Deep Learning',
            year='2023',
            source_type='paper',
            id='paper_102',
            text_summary='Deep learning methods for protein structure prediction.',
            text='Several neural network architectures are compared for protein structure prediction.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Transformer Models for Text Classification',
            year='2024',
            source_type='paper',
            id='paper_103',
            text_summary='Transformer architectures for natural language classification.',
            text='This paper studies transformer models for text classification.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Protein Folding: A Historical Review',
            year='2010',
            source_type='paper',
            id='paper_104',
            text_summary='Historical development of computational protein folding.',
            text='The paper discusses classical approaches to protein folding.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 4,
     'first_id': 'paper_101',
     'contains_ids': ['paper_101', 'paper_102', 'paper_104'],
     'order_before': [('paper_101', 'paper_102'), ('paper_102', 'paper_104')]}
,
}

# ------------------------------------------------------------
# test_case_3
# ------------------------------------------------------------
test_case_3 = {
    "name": 'case_3_none_fields',
    "query": 'What is reinforcement learning?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Introduction to Reinforcement Learning',
            year='2022',
            source_type='paper',
            id='paper_201',
            text_summary='An introduction to reinforcement learning concepts.',
            text='Reinforcement learning is a machine learning paradigm based on rewards and actions.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title=None,
            year=None,
            source_type=None,
            id='paper_202',
            text_summary=None,
            text=None,
            image_summary=None,
            image_base_64=None,
            evidence_level=None,
            last_retrieved_at=None,
        ),
        Metadata(
            title='Deep Reinforcement Learning',
            year='2023',
            source_type='paper',
            id='paper_203',
            text_summary=None,
            text='Deep neural networks can be combined with reinforcement learning.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title=None,
            year='2020',
            source_type='paper',
            id='paper_204',
            text_summary='A study about reinforcement learning algorithms.',
            text=None,
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'contains_ids': ['paper_201', 'paper_203'],
     'excludes_ids': ['paper_202'],
     'order_before': [('paper_201', 'paper_203')]}
,
}

# ------------------------------------------------------------
# test_case_4
# ------------------------------------------------------------
test_case_4 = {
    "name": 'case_4_alzheimer',
    "query": "What methods are used for detecting Alzheimer's disease?",
    "k1": 3,
    "metadatas": [
        Metadata(
            title="Machine Learning for Alzheimer's Disease Detection",
            year='2024',
            source_type='paper',
            id='paper_301',
            text_summary="Machine learning methods for detecting Alzheimer's disease using MRI data.",
            text="This paper proposes machine learning methods to detect Alzheimer's disease from MRI scans.",
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title="MRI-Based Diagnosis of Alzheimer's Disease",
            year='2023',
            source_type='paper',
            id='paper_302',
            text_summary="MRI-based approaches for Alzheimer's disease diagnosis.",
            text="MRI data is analyzed to identify biomarkers associated with Alzheimer's disease.",
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Machine Learning in Radiology',
            year='2024',
            source_type='paper',
            id='paper_303',
            text_summary='Applications of machine learning to radiology.',
            text='Machine learning is increasingly used in medical image analysis.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_301',
     'contains_ids': ['paper_301', 'paper_302', 'paper_303'],
     'order_before': [('paper_301', 'paper_302'), ('paper_302', 'paper_303')]}
,
}

# ------------------------------------------------------------
# test_case_5
# ------------------------------------------------------------
test_case_5 = {
    "name": 'case_5_covid_vaccine',
    "query": 'What evidence exists for the effectiveness of COVID-19 vaccines?',
    "k1": 4,
    "metadatas": [
        Metadata(
            title='COVID-19 Vaccine Effectiveness: A Systematic Review',
            year='2024',
            source_type='systematic_review',
            id='paper_401',
            text_summary='Systematic review of clinical and observational evidence on COVID-19 vaccine effectiveness.',
            text='We systematically review evidence regarding vaccine effectiveness.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='COVID-19 Vaccine Clinical Trial',
            year='2021',
            source_type='clinical_trial',
            id='paper_402',
            text_summary='Randomized clinical trial evaluating COVID-19 vaccine efficacy.',
            text='A randomized controlled trial evaluated vaccine efficacy against symptomatic infection.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='COVID-19 Vaccine News',
            year='2024',
            source_type='news',
            id='paper_403',
            text_summary='News report discussing COVID-19 vaccination.',
            text='The article reports recent developments concerning COVID-19 vaccines.',
            image_summary=None,
            image_base_64=None,
            evidence_level='low',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='COVID-19 Vaccine Wikipedia Article',
            year='2025',
            source_type='webpage',
            id='paper_404',
            text_summary='Overview of COVID-19 vaccines and their development.',
            text='This page summarizes information about COVID-19 vaccines.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 4,
     'first_id': 'paper_401',
     'contains_ids': ['paper_401', 'paper_402', 'paper_403', 'paper_404'],
     'order_before': [('paper_401', 'paper_402'), ('paper_402', 'paper_403')]}
,
}

# ------------------------------------------------------------
# test_case_6
# ------------------------------------------------------------
test_case_6 = {
    "name": 'case_6_smoking_lung_cancer',
    "query": 'Does smoking increase the risk of lung cancer?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Smoking and Lung Cancer Risk: Meta-analysis',
            year='2024',
            source_type='meta_analysis',
            id='paper_501',
            text_summary='Meta-analysis examining the association between smoking and lung cancer.',
            text='The meta-analysis finds a strong association between cigarette smoking and lung cancer risk.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Smoking and Cancer',
            year='2023',
            source_type='paper',
            id='paper_502',
            text_summary='A review of smoking-related cancers.',
            text='Smoking is associated with several types of cancer.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Smoking May Cause Lung Cancer',
            year='2025',
            source_type='blog',
            id='paper_503',
            text_summary='A blog post discussing possible health risks of smoking.',
            text='Smoking may increase the risk of developing lung cancer.',
            image_summary=None,
            image_base_64=None,
            evidence_level='low',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_501',
     'contains_ids': ['paper_501', 'paper_502', 'paper_503'],
     'order_before': [('paper_501', 'paper_502'), ('paper_502', 'paper_503')]}
,
}

# ------------------------------------------------------------
# test_case_7
# ------------------------------------------------------------
test_case_7 = {
    "name": 'case_7_cnn_image_summary',
    "query": 'What does a convolutional neural network architecture look like?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Convolutional Neural Networks',
            year='2022',
            source_type='paper',
            id='paper_601',
            text_summary='Introduction to CNN architectures.',
            text='The paper describes convolution, pooling and fully connected layers.',
            image_summary=['A diagram showing convolutional neural network architecture with convolution and pooling layers.'],
            image_base_64=['base64_image_1'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Deep Learning Fundamentals',
            year='2021',
            source_type='paper',
            id='paper_602',
            text_summary='Overview of fundamental deep learning concepts.',
            text='The paper introduces neural networks and optimization.',
            image_summary=['A diagram showing a generic neural network.'],
            image_base_64=['base64_image_2'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='CNN Visualization',
            year='2023',
            source_type='webpage',
            id='paper_603',
            text_summary='Visual explanation of convolutional neural networks.',
            text='This webpage explains CNN architectures using diagrams.',
            image_summary=['Detailed visualization of convolutional and pooling layers.'],
            image_base_64=['base64_image_3'],
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_601',
     'contains_ids': ['paper_601', 'paper_602', 'paper_603'],
     'order_before': [('paper_601', 'paper_603'), ('paper_603', 'paper_602')]}
,
}

# ------------------------------------------------------------
# test_case_8
# ------------------------------------------------------------
test_case_8 = {
    "name": 'case_8_irrelevant_noise',
    "query": 'How does climate change affect coral reefs?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Climate Change and Coral Reef Ecosystems',
            year='2024',
            source_type='paper',
            id='paper_701',
            text_summary='Effects of increasing ocean temperature and acidification on coral reefs.',
            text='Climate change causes coral bleaching and changes in marine ecosystems.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Ocean Acidification and Marine Life',
            year='2023',
            source_type='paper',
            id='paper_702',
            text_summary='Effects of ocean acidification on marine organisms.',
            text='Ocean acidification affects marine organisms and ecosystem stability.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Transformer Architecture',
            year='2024',
            source_type='paper',
            id='paper_703',
            text_summary='Transformer models for natural language processing.',
            text='This paper introduces an efficient transformer architecture.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Quantum Computing Algorithms',
            year='2022',
            source_type='paper',
            id='paper_704',
            text_summary='Algorithms for quantum computers.',
            text='The paper studies quantum search and optimization algorithms.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_701',
     'contains_ids': ['paper_701', 'paper_702', 'paper_703'],
     'excludes_ids': ['paper_704'],
     'order_before': [('paper_701', 'paper_702'), ('paper_702', 'paper_703')]}
,
}

# ------------------------------------------------------------
# test_case_9
# ------------------------------------------------------------
test_case_9 = {
    "name": 'case_9_near_duplicates',
    "query": 'What are the applications of large language models in education?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Large Language Models in Education',
            year='2024',
            source_type='paper',
            id='paper_801',
            text_summary='Applications of large language models for personalized education and automated feedback.',
            text='Large language models can support personalized learning and automated feedback.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Large Language Models for Educational Applications',
            year='2024',
            source_type='paper',
            id='paper_802',
            text_summary='Applications of LLMs in personalized learning and education.',
            text='LLMs can provide personalized learning assistance and automated feedback.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Generative AI in Education',
            year='2023',
            source_type='paper',
            id='paper_803',
            text_summary='A broader survey of generative AI applications in education.',
            text='Generative AI can be used to create educational materials and support teachers.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'contains_ids': ['paper_801', 'paper_802', 'paper_803'],
     'order_before': [('paper_801', 'paper_803')]}
,
}

# ------------------------------------------------------------
# test_case_10
# ------------------------------------------------------------
test_case_10 = {
    "name": 'case_10_k1_gt_len',
    "query": 'What is federated learning?',
    "k1": 10,
    "metadatas": [
        Metadata(
            title='Federated Learning',
            year='2023',
            source_type='paper',
            id='paper_901',
            text_summary='Introduction to federated learning.',
            text='Federated learning enables decentralized machine learning without sharing raw data.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Distributed Machine Learning',
            year='2022',
            source_type='paper',
            id='paper_902',
            text_summary='Distributed approaches to machine learning.',
            text='Machine learning can be distributed across multiple computational nodes.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 2,
     'first_id': 'paper_901',
     'contains_ids': ['paper_901', 'paper_902'],
     'order_before': [('paper_901', 'paper_902')]}
,
}

# ------------------------------------------------------------
# test_case_11
# ------------------------------------------------------------
test_case_11 = {
    "name": 'case_11_empty',
    "query": 'What is quantum machine learning?',
    "k1": 5,
    "metadatas": [
    ],
    "expect": {'len': 0}
,
}

# ------------------------------------------------------------
# test_case_12
# ------------------------------------------------------------
test_case_12 = {
    "name": 'case_12_k1_zero',
    "query": 'What is machine learning?',
    "k1": 0,
    "metadatas": [
        Metadata(
            title='Introduction to Machine Learning',
            year='2023',
            source_type='paper',
            id='paper_1001',
            text_summary='An introduction to machine learning.',
            text='Machine learning is a field of artificial intelligence.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 0}
,
}

# ------------------------------------------------------------
# test_case_13
# ------------------------------------------------------------
test_case_13 = {
    "name": 'case_13_empty_image_lists',
    "query": 'What is graph neural network message passing?',
    "k1": 2,
    "metadatas": [
        Metadata(
            title='Graph Neural Networks and Message Passing',
            year='2024',
            source_type='paper',
            id='paper_1101',
            text_summary='Message passing frameworks for graph neural networks.',
            text='We study message passing neural networks on graph-structured data.',
            image_summary=[],
            image_base_64=[],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Introduction to Convolutional Networks',
            year='2020',
            source_type='paper',
            id='paper_1102',
            text_summary='CNNs for image classification.',
            text='Convolutional layers extract spatial features from images.',
            image_summary=[],
            image_base_64=[],
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Message Passing on Graphs',
            year='2023',
            source_type='paper',
            id='paper_1103',
            text_summary=None,
            text=None,
            image_summary=[],
            image_base_64=[],
            evidence_level='low',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 2,
     'first_id': 'paper_1101',
     'contains_ids': ['paper_1101', 'paper_1103'],
     'excludes_ids': ['paper_1102'],
     'order_before': [('paper_1101', 'paper_1103')]}
,
}

# ------------------------------------------------------------
# test_case_14
# ------------------------------------------------------------
test_case_14 = {
    "name": 'case_14_multi_image_summaries',
    "query": 'How do attention heatmaps and architecture diagrams explain transformers?',
    "k1": 2,
    "metadatas": [
        Metadata(
            title='Visualizing Transformers',
            year='2024',
            source_type='paper',
            id='paper_1201',
            text_summary='A short note on transformer visualization.',
            text='Figures help explain model internals.',
            image_summary=['Attention heatmap showing token-to-token attention weights.', 'Architecture diagram of multi-head self-attention blocks.', 'Training loss curve for the transformer model.'],
            image_base_64=['img_attn_heatmap', 'img_arch_diagram', 'img_loss_curve'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Transformer Overview',
            year='2023',
            source_type='paper',
            id='paper_1202',
            text_summary='High-level overview of transformers without visual analysis.',
            text='Transformers use self-attention for sequence modeling.',
            image_summary=['A generic neural network sketch unrelated to attention heatmaps.'],
            image_base_64=['img_generic'],
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='CNN Feature Map Gallery',
            year='2022',
            source_type='webpage',
            id='paper_1203',
            text_summary='Collection of convolutional feature maps.',
            text='Feature maps illustrate CNN activations.',
            image_summary=['Convolutional feature map of an edge detector.', 'Pooling layer output visualization.'],
            image_base_64=['img_feat1', 'img_feat2'],
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 2,
     'first_id': 'paper_1201',
     'contains_ids': ['paper_1201'],
     'excludes_ids': ['paper_1203'],
     'order_before': [('paper_1201', 'paper_1202')]}
,
}

ROUND1_CASES = [
    test_case_1,
    test_case_2,
    test_case_3,
    test_case_4,
    test_case_5,
    test_case_6,
    test_case_7,
    test_case_8,
    test_case_9,
    test_case_10,
    test_case_11,
    test_case_12,
    test_case_13,
    test_case_14,
]
