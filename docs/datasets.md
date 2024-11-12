# Datasets

> [!NOTE]  
> UnsafeQA, PromptsDE, PromptsFR, PromptsIT, and PromptsES will be released soon!

| Dataset                                                 | Category      | Sub-category | Total | Unsafe | Alias                      |
| :------------------------------------------------------ | :------------ | :----------- | ----: | -----: | :------------------------- |
| [AdvBench Behaviors](#advbench-behaviors)               | Prompts       | Instructions |   520 |   100% | advbench_behaviors         |
| [HarmBench Behaviors](#harmbench-behaviors)             | Prompts       | Instructions |   320 |   100% | harmbench_behaviors        |
| [I-CoNa](#i-cona)                                       | Prompts       | Instructions |   178 |   100% | i_cona                     |
| [I-Controversial](#i-controversial)                     | Prompts       | Instructions |    40 |   100% | i_controversial            |
| [I-MaliciousInstructions](#i-maliciousinstructions)     | Prompts       | Instructions |   100 |   100% | i_malicious_instructions   |
| [I-Physical-Safety](#i-physical-safety)                 | Prompts       | Instructions |   200 |    50% | i_physical_safety          |
| [MaliciousInstruct](#maliciousinstruct)                 | Prompts       | Instructions |   100 |   100% | malicious_instruct         |
| [MITRE](#mitre)                                         | Prompts       | Instructions |   977 |   100% | mitre                      |
| [StrongREJECT Instructions](#strongreject-instructions) | Prompts       | Instructions |   213 |   100% | strong_reject_instructions |
| [TDCRedTeaming](#tdcredteaming)                         | Prompts       | Instructions |    50 |   100% | tdc_red_teaming            |
| [CatQA](#catqa)                                         | Prompts       | Questions    |   550 |   100% | cat_qa                     |
| [Do Anything Now Questions](#do-anything-now-questions) | Prompts       | Questions    |   390 |   100% | do_anything_now_questions  |
| [DoNotAnswer](#donotanswer)                             | Prompts       | Questions    |   939 |   100% | do_not_answer              |
| [HarmfulQ](#harmfulq)                                   | Prompts       | Questions    |   200 |   100% | harmful_q                  |
| [HarmfulQA Questions](#harmfulqa-questions)             | Prompts       | Questions    |  1960 |   100% | harmful_qa_questions       |
| [HEx-PHI](#hex-phi)                                     | Prompts       | Questions    |   330 |   100% | hex_phi                    |
| [XSTest](#xstest)                                       | Prompts       | Questions    |   450 |    44% | xstest                     |
| [AdvBench Strings](#advbench-strings)                   | Prompts       | Statements   |   574 |   100% | advbench_strings           |
| [DecodingTrust Stereotypes](#decodingtrust-stereotypes) | Prompts       | Statements   |  1152 |   100% | decoding_trust_stereotypes |
| [DynaHate](#dynahate)                                   | Prompts       | Statements   |  4120 |    55% | dynahate                   |
| [HateCheck](#hatecheck)                                 | Prompts       | Statements   |  3728 |    69% | hatecheck                  |
| [Hatemoji Check](#hatemoji-check)                       | Prompts       | Statements   |  3683 |    71% | hatemoji_check             |
| [SafeText](#safetext)                                   | Prompts       | Statements   |  1465 |    25% | safe_text                  |
| [ToxiGen](#toxigen)                                     | Prompts       | Statements   |   940 |    43% | toxigen                    |
| [AART](#aart)                                           | Prompts       | Mixed        |  3269 |   100% | aart                       |
| [OpenAI Moderation Dataset](#openai-moderation-dataset) | Prompts       | Mixed        |  1680 |    31% | openai_moderation_dataset  |
| [SimpleSafetyTests](#simplesafetytests)                 | Prompts       | Mixed        |   100 |   100% | simple_safety_tests        |
| [Toxic Chat](#toxic-chat)                               | Prompts       | Mixed        |  5083 |     7% | toxic_chat                 |
| [BeaverTails 330k](#beavertails-330k)                   | Conversations | Single-Turn  | 11088 |    55% | beaver_tails_330k          |
| [Bot-Adversarial Dialogue](#bot-adversarial-dialogue)   | Conversations | Multi-Turn   |  2598 |    36% | bot_adversarial_dialogue   |
| [ConvAbuse](#convabuse)                                 | Conversations | Multi-Turn   |   853 |    15% | convabuse                  |
| [DICES 350](#dices-350)                                 | Conversations | Multi-Turn   |   350 |    23% | dices_350                  |
| [DICES 990](#dices-990)                                 | Conversations | Multi-Turn   |   990 |    16% | dices_990                  |
| [HarmfulQA](#harmfulqa)                                 | Conversations | Multi-Turn   | 16459 |    45% | harmful_qa                 |
| [ProsocialDialog](#prosocialdialog)                     | Conversations | Multi-Turn   | 25029 |    60% | prosocial_dialog           |

## AdvBench Behaviors

- Language(s): English
- Source(s): Machine-generated
- License: `MIT`
- Alias: `advbench_behaviors`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{advbench,
    author       = {Andy Zou and
                    Zifan Wang and
                    J. Zico Kolter and
                    Matt Fredrikson},
    title        = {Universal and Transferable Adversarial Attacks on Aligned Language
                    Models},
    journal      = {CoRR},
    volume       = {abs/2307.15043},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2307.15043},
    doi          = {10.48550/ARXIV.2307.15043},
    eprinttype    = {arXiv},
    eprint       = {2307.15043},
    timestamp    = {Mon, 28 Aug 2023 21:26:19 +0200},
  }
  ```
</details>

## HarmBench Behaviors

- Language(s): English
- Source(s): Human-generated
- License: `MIT`
- Alias: `harmbench_behaviors`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{harmbench_behaviors,
    author       = {Mantas Mazeika and
                    Long Phan and
                    Xuwang Yin and
                    Andy Zou and
                    Zifan Wang and
                    Norman Mu and
                    Elham Sakhaee and
                    Nathaniel Li and
                    Steven Basart and
                    Bo Li and
                    David A. Forsyth and
                    Dan Hendrycks},
    title        = {HarmBench: {A} Standardized Evaluation Framework for Automated Red
                    Teaming and Robust Refusal},
    journal      = {CoRR},
    volume       = {abs/2402.04249},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2402.04249},
    doi          = {10.48550/ARXIV.2402.04249},
    eprinttype    = {arXiv},
    eprint       = {2402.04249},
    timestamp    = {Mon, 12 Feb 2024 17:35:51 +0100},
  }
  ```
</details>

## I-CoNa

- Language(s): English
- Source(s): Human-generated
- License: `CC BY-NC 4.0`
- Alias: `i_cona`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{i_datasets,
    author       = {Federico Bianchi and
                    Mirac Suzgun and
                    Giuseppe Attanasio and
                    Paul R{"{o}}ttger and
                    Dan Jurafsky and
                    Tatsunori Hashimoto and
                    James Zou},
    title        = {Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language
                    Models that Follow Instructions},
    journal      = {CoRR},
    volume       = {abs/2309.07875},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2309.07875},
    doi          = {10.48550/ARXIV.2309.07875},
    eprinttype    = {arXiv},
    eprint       = {2309.07875},
    timestamp    = {Wed, 20 Sep 2023 11:04:35 +0200},
  }
  ```
</details>

## I-Controversial

- Language(s): English
- Source(s): Human-generated
- License: `CC BY-NC 4.0`
- Alias: `i_controversial`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{i_datasets,
    author       = {Federico Bianchi and
                    Mirac Suzgun and
                    Giuseppe Attanasio and
                    Paul R{"{o}}ttger and
                    Dan Jurafsky and
                    Tatsunori Hashimoto and
                    James Zou},
    title        = {Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language
                    Models that Follow Instructions},
    journal      = {CoRR},
    volume       = {abs/2309.07875},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2309.07875},
    doi          = {10.48550/ARXIV.2309.07875},
    eprinttype    = {arXiv},
    eprint       = {2309.07875},
    timestamp    = {Wed, 20 Sep 2023 11:04:35 +0200},
  }
  ```
</details>

## I-MaliciousInstructions

- Language(s): English
- Source(s): Machine-generated
- License: `CC BY-NC 4.0`
- Alias: `i_malicious_instructions`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{i_datasets,
    author       = {Federico Bianchi and
                    Mirac Suzgun and
                    Giuseppe Attanasio and
                    Paul R{"{o}}ttger and
                    Dan Jurafsky and
                    Tatsunori Hashimoto and
                    James Zou},
    title        = {Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language
                    Models that Follow Instructions},
    journal      = {CoRR},
    volume       = {abs/2309.07875},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2309.07875},
    doi          = {10.48550/ARXIV.2309.07875},
    eprinttype    = {arXiv},
    eprint       = {2309.07875},
    timestamp    = {Wed, 20 Sep 2023 11:04:35 +0200},
  }
  ```
</details>

## I-Physical-Safety

- Language(s): English
- Source(s): Human-generated
- License: `CC BY-NC 4.0`
- Alias: `i_physical_safety`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{i_datasets,
    author       = {Federico Bianchi and
                    Mirac Suzgun and
                    Giuseppe Attanasio and
                    Paul R{"{o}}ttger and
                    Dan Jurafsky and
                    Tatsunori Hashimoto and
                    James Zou},
    title        = {Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language
                    Models that Follow Instructions},
    journal      = {CoRR},
    volume       = {abs/2309.07875},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2309.07875},
    doi          = {10.48550/ARXIV.2309.07875},
    eprinttype    = {arXiv},
    eprint       = {2309.07875},
    timestamp    = {Wed, 20 Sep 2023 11:04:35 +0200},
  }
  ```
</details>

## MaliciousInstruct

- Language(s): English
- Source(s): Machine-generated
- License: `MIT`
- Alias: `malicious_instruct`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{malicious_instruct,
    author       = {Yangsibo Huang and
                    Samyak Gupta and
                    Mengzhou Xia and
                    Kai Li and
                    Danqi Chen},
    title        = {Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation},
    journal      = {CoRR},
    volume       = {abs/2310.06987},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2310.06987},
    doi          = {10.48550/ARXIV.2310.06987},
    eprinttype    = {arXiv},
    eprint       = {2310.06987},
    timestamp    = {Thu, 14 Dec 2023 18:03:42 +0100},
  }
  ```
</details>

## MITRE

- Language(s): English
- Source(s): Human-generated
- License: `MIT`
- Alias: `mitre`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{mitre,
    author       = {Manish Bhatt and
                    Sahana Chennabasappa and
                    Yue Li and
                    Cyrus Nikolaidis and
                    Daniel Song and
                    Shengye Wan and
                    Faizan Ahmad and
                    Cornelius Aschermann and
                    Yaohui Chen and
                    Dhaval Kapil and
                    David Molnar and
                    Spencer Whitman and
                    Joshua Saxe},
    title        = {CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models},
    journal      = {CoRR},
    volume       = {abs/2404.13161},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2404.13161},
    doi          = {10.48550/ARXIV.2404.13161},
    eprinttype    = {arXiv},
    eprint       = {2404.13161},
    timestamp    = {Fri, 19 Apr 2024 11:04:35 +0200},
  }
  ```
</details>

## StrongREJECT Instructions

- Language(s): English
- Source(s): Human-generated
- License: `N/A`
- Alias: `strong_reject_instructions`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{strong_reject_instructions,
    author       = {Alexandra Souly and
                    Qingyuan Lu and
                    Dillon Bowen and
                    Tu Trinh and
                    Elvis Hsieh and
                    Sana Pandey and
                    Pieter Abbeel and
                    Justin Svegliato and
                    Scott Emmons and
                    Olivia Watkins and
                    Sam Toyer},
    title        = {A StrongREJECT for Empty Jailbreaks},
    journal      = {CoRR},
    volume       = {abs/2402.10260},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2402.
  10260},
    doi          = {10.48550/ARXIV.2402.10260},
    eprinttype   = {arXiv},
    eprint       = {2402.10260},
    timestamp    = {Mon, 26 Feb 2024 16:52:48 +0100},
  }
  ```
</details>

## TDCRedTeaming

- Language(s): English
- Source(s): Human-generated
- License: `MIT`
- Alias: `tdc_red_teaming`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{tdc_red_teaming,
    author       = {Mantas Mazeika and
                    Andy Zou and
                    Norman Mu and
                    Long Phan and
                    Zifan Wang and
                    Chunru Yu and
                    Adam Khoja and
                    Fengqing Jiang and
                    Aidan O'Gara and
                    Ellie Sakhaee and
                    Zhen Xiang and
                    Arezoo Rajabi and
                    Dan Hendrycks and
                    Radha Poovendran and
                    Bo Li and
                    David Forsyth},
    title        = {TDC 2023 (LLM Edition): The Trojan Detection 
  Challenge},
    booktitle    = {NeurIPS Competition Track},
    year         = {2023},
  }
  ```
</details>

## CatQA

- Language(s): English
- Source(s): Machine-generated
- License: `Apache 2.0`
- Alias: `cat_qa`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{cat_qa,
    author       = {Rishabh Bhardwaj and
                    Do Duc Anh and
                    Soujanya Poria},
    title        = {Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned
                    Language Models through Task Arithmetic},
    journal      = {CoRR},
    volume       = {abs/2402.11746},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2402.11746},
    doi          = {10.48550/ARXIV.2402.11746},
    eprinttype    = {arXiv},
    eprint       = {2402.11746},
    timestamp    = {Mon, 26 Feb 2024 16:52:48 +0100},
  }
  ```
</details>

## Do Anything Now Questions

- Language(s): English
- Source(s): Discord, Reddit, AIPRM, FlowGPT, JailbreakChat, Awesome ChatGPT Prompts, OCR-Prompts
- License: `MIT`
- Alias: `do_anything_now_questions`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{do_anything_now,
    author       = {Xinyue Shen and
                    Zeyuan Chen and
                    Michael Backes and
                    Yun Shen and
                    Yang Zhang},
    title        = {"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak
                    Prompts on Large Language Models},
    journal      = {CoRR},
    volume       = {abs/2308.03825},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2308.03825},
    doi          = {10.48550/ARXIV.2308.03825},
    eprinttype    = {arXiv},
    eprint       = {2308.03825},
    timestamp    = {Fri, 27 Oct 2023 08:36:05 +0200},
  }
  ```
</details>

## DoNotAnswer

- Language(s): English
- Source(s): Machine-generated
- License: `Apache 2.0`
- Alias: `do_not_answer`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{do_not_answer,
    author       = {Yuxia Wang and
                    Haonan Li and
                    Xudong Han and
                    Preslav Nakov and
                    Timothy Baldwin},
    title        = {Do-Not-Answer: Evaluating Safeguards in LLMs},
    booktitle    = {Findings of the Association for Computational Linguistics: {EACL}
                    2024, St. Julian's, Malta, March 17-22, 2024},
    pages        = {896--911},
    publisher    = {Association for Computational Linguistics},
    year         = {2024},
    url          = {https://aclanthology.org/2024.findings-eacl.61},
    timestamp    = {Tue, 02 Apr 2024 16:32:10 +0200},
  }
  ```
</details>

## HarmfulQ

- Language(s): English
- Source(s): Machine-generated
- License: `CC BY SA 4.0`
- Alias: `harmful_q`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{harmful_q,
    author       = {Omar Shaikh and
                    Hongxin Zhang and
                    William Held and
                    Michael S. Bernstein and
                    Diyi Yang},
    title        = {On Second Thought, Let's Not Think Step by Step! Bias and Toxicity
                    in Zero-Shot Reasoning},
    booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                    Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                    July 9-14, 2023},
    pages        = {4454--4470},
    publisher    = {Association for Computational Linguistics},
    year         = {2023},
    url          = {https://doi.org/10.18653/v1/2023.acl-long.244},
    doi          = {10.18653/V1/2023.ACL-LONG.244},
    timestamp    = {Mon, 25 Sep 2023 15:32:28 +0200},
  }
  ```
</details>

## HarmfulQA Questions

- Language(s): English
- Source(s): Machine-generated
- License: `Apache 2.0`
- Alias: `harmful_qa_questions`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{harmful_qa,
    author       = {Rishabh Bhardwaj and
                    Soujanya Poria},
    title        = {Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment},
    journal      = {CoRR},
    volume       = {abs/2308.09662},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2308.09662},
    doi          = {10.48550/ARXIV.2308.09662},
    eprinttype    = {arXiv},
    eprint       = {2308.09662},
    timestamp    = {Fri, 25 Aug 2023 11:24:49 +0200},
  }
  ```
</details>

## HEx-PHI

- Language(s): English
- Source(s): Human-generated
- License: `https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI#hex-phi-dataset-license-agreement`
- Alias: `hex_phi`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{hex_phi,
    author       = {Xiangyu Qi and
                    Yi Zeng and
                    Tinghao Xie and
                    Pin{-}Yu Chen and
                    Ruoxi Jia and
                    Prateek Mittal and
                    Peter Henderson},
    title        = {Fine-tuning Aligned Language Models Compromises Safety, Even When
                    Users Do Not Intend To!},
    journal      = {CoRR},
    volume       = {abs/2310.03693},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2310.03693},
    doi          = {10.48550/ARXIV.2310.03693},
    eprinttype    = {arXiv},
    eprint       = {2310.03693},
    timestamp    = {Wed, 20 Mar 2024 11:34:12 +0100},
  }
  ```
</details>

## XSTest

- Language(s): English
- Source(s): Human-generated
- License: `CC BY 4.0`
- Alias: `xstest`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{xstest,
    author       = {Paul R{"{o}}ttger and
                    Hannah Rose Kirk and
                    Bertie Vidgen and
                    Giuseppe Attanasio and
                    Federico Bianchi and
                    Dirk Hovy},
    title        = {XSTest: {A} Test Suite for Identifying Exaggerated Safety Behaviours
                    in Large Language Models},
    journal      = {CoRR},
    volume       = {abs/2308.01263},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2308.01263},
    doi          = {10.48550/ARXIV.2308.01263},
    eprinttype   = {arXiv},
    eprint       = {2308.01263},
    timestamp    = {Mon, 21 Aug 2023 17:38:10 +0200},
  }
  ```
</details>

## AdvBench Strings

- Language(s): English
- Source(s): Machine-generated
- License: `MIT`
- Alias: `advbench_strings`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{advbench,
    author       = {Andy Zou and
                    Zifan Wang and
                    J. Zico Kolter and
                    Matt Fredrikson},
    title        = {Universal and Transferable Adversarial Attacks on Aligned Language
                    Models},
    journal      = {CoRR},
    volume       = {abs/2307.15043},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2307.15043},
    doi          = {10.48550/ARXIV.2307.15043},
    eprinttype    = {arXiv},
    eprint       = {2307.15043},
    timestamp    = {Mon, 28 Aug 2023 21:26:19 +0200},
  }
  ```
</details>

## DecodingTrust Stereotypes

- Language(s): English
- Source(s): Human-generated
- License: `CC BY SA 4.0`
- Alias: `decoding_trust_stereotypes`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{decoding_trust_stereotypes,
    author       = {Boxin Wang and
                    Weixin Chen and
                    Hengzhi Pei and
                    Chulin Xie and
                    Mintong Kang and
                    Chenhui Zhang and
                    Chejian Xu and
                    Zidi Xiong and
                    Ritik Dutta and
                    Rylan Schaeffer and
                    Sang T. Truong and
                    Simran Arora and
                    Mantas Mazeika and
                    Dan Hendrycks and
                    Zinan Lin and
                    Yu Cheng and
                    Sanmi Koyejo and
                    Dawn Song and
                    Bo Li},
    title        = {DecodingTrust: {A} Comprehensive Assessment of Trustworthiness in
                    {GPT} Models},
    booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                    on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                    LA, USA, December 10 - 16, 2023},
    year         = {2023},
    url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/
  63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets\_and\_Benchmarks.html},
    timestamp    = {Tue, 02 Apr 2024 16:32:39 +0200},
  }
  ```
</details>

## DynaHate

- Language(s): English
- Source(s): Human-generated
- License: `Apache 2.0`
- Alias: `dynahate`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{dynahate,
    author       = {Bertie Vidgen and
                    Tristan Thrush and
                    Zeerak Waseem and
                    Douwe Kiela},
    title        = {Learning from the Worst: Dynamically Generated Datasets to Improve
                    Online Hate Detection},
    booktitle    = {Proceedings of the 59th Annual Meeting of the Association for Computational
                    Linguistics and the 11th International Joint Conference on Natural
                    Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
                    Event, August 1-6, 2021},
    pages        = {1667--1682},
    publisher    = {Association for Computational Linguistics},
    year         = {2021},
    url          = {https://doi.org/10.18653/v1/2021.acl-long.132},
    doi          = {10.18653/V1/2021.ACL-LONG.132},
    timestamp    = {Mon, 09 Aug 2021 16:25:37 +0200},
  }
  ```
</details>

## HateCheck

- Language(s): English
- Source(s): Algorithmically generated
- License: `CC BY 4.0`
- Alias: `hatecheck`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{hatecheck,
    author       = {Paul R{"{o}}ttger and
                    Bertie Vidgen and
                    Dong Nguyen and
                    Zeerak Waseem and
                    Helen Z. Margetts and
                    Janet B. Pierrehumbert},
    title        = {HateCheck: Functional Tests for Hate Speech Detection Models},
    booktitle    = {Proceedings of the 59th Annual Meeting of the Association for Computational
                    Linguistics and the 11th International Joint Conference on Natural
                    Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
                    Event, August 1-6, 2021},
    pages        = {41--58},
    publisher    = {Association for Computational Linguistics},
    year         = {2021},
    url          = {https://doi.org/10.18653/v1/2021.acl-long.4},
    doi          = {10.18653/V1/2021.ACL-LONG.4},
    timestamp    = {Mon, 05 Feb 2024 20:27:00 +0100},
  }
  ```
</details>

## Hatemoji Check

- Language(s): English
- Source(s): Algorithmically generated
- License: `CC BY 4.0`
- Alias: `hatemoji_check`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{hatemoji,
    author       = {Hannah Kirk and
                    Bertie Vidgen and
                    Paul R{"{o}}ttger and
                    Tristan Thrush and
                    Scott A. Hale},
    title        = {Hatemoji: {A} Test Suite and Adversarially-Generated Dataset for Benchmarking
                    and Detecting Emoji-Based Hate},
    booktitle    = {Proceedings of the 2022 Conference of the North American Chapter of
                    the Association for Computational Linguistics: Human Language Technologies,
                    {NAACL} 2022, Seattle, WA, United States, July 10-15, 2022},
    pages        = {1352--1368},
    publisher    = {Association for Computational Linguistics},
    year         = {2022},
    url          = {https://doi.org/10.18653/v1/2022.naacl-main.97},
    doi          = {10.18653/V1/2022.NAACL-MAIN.97},
    timestamp    = {Mon, 26 Jun 2023 20:46:58 +0200},
  }
  ```
</details>

## SafeText

- Language(s): English
- Source(s): Human-generated
- License: `MIT`
- Alias: `safe_text`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{safe_text,
    author       = {Sharon Levy and
                    Emily Allaway and
                    Melanie Subbiah and
                    Lydia B. Chilton and
                    Desmond Patton and
                    Kathleen R. McKeown and
                    William Yang Wang},
    title        = {SafeText: {A} Benchmark for Exploring Physical Safety in Language
                    Models},
    booktitle    = {Proceedings of the 2022 Conference on Empirical Methods in Natural
                    Language Processing, {EMNLP} 2022, Abu Dhabi, United Arab Emirates,
                    December 7-11, 2022},
    pages        = {2407--2421},
    publisher    = {Association for Computational Linguistics},
    year         = {2022},
    url          = {https://doi.org/10.18653/v1/2022.emnlp-main.154},
    doi          = {10.18653/V1/2022.EMNLP-MAIN.154},
    timestamp    = {Thu, 10 Aug 2023 12:35:27 +0200},
  }
  ```
</details>

## ToxiGen

- Language(s): English
- Source(s): Machine-generated
- License: `MIT`
- Alias: `toxigen`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{toxigen,
    title        = "{T}oxi{G}en: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection",
    author       = "Hartvigsen, Thomas  and
                   Gabriel, Saadia  and
                   Palangi, Hamid  and
                   Sap, Maarten  and
                   Ray, Dipankar  and
                   Kamar, Ece",
    editor       = "Muresan, Smaranda  and
                    Nakov, Preslav  and
                    Villavicencio, Aline",
    booktitle    = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month        = may,
    year         = "2022",
    address      = "Dublin, Ireland",
    publisher    = "Association for Computational Linguistics",
    url          = "https://aclanthology.org/2022.acl-long.234",
    doi          = "10.18653/v1/2022.acl-long.234",
    pages        = "3309--3326",
  }
  ```
</details>

## AART

- Language(s): English
- Source(s): Machine-generated
- License: `CC BY 4.0`
- Alias: `aart`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{aart,
    author       = {Bhaktipriya Radharapu and
                    Kevin Robinson and
                    Lora Aroyo and
                    Preethi Lahoti},
    title        = {{AART:} AI-Assisted Red-Teaming with Diverse Data Generation for New
                    LLM-powered Applications},
    booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                    Language Processing: {EMNLP} 2023 - Industry Track, Singapore, December
                    6-10, 2023},
    pages        = {380--395},
    publisher    = {Association for Computational Linguistics},
    year         = {2023},
    url          = {https://doi.org/10.18653/v1/2023.emnlp-industry.37},
    doi          = {10.18653/V1/2023.EMNLP-INDUSTRY.37},
    timestamp    = {Fri, 12 Apr 2024 13:11:30 +0200},
  }
  ```
</details>

## OpenAI Moderation Dataset

- Language(s): English
- Source(s): Human-generated
- License: `MIT`
- Alias: `openai_moderation_dataset`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{openai_moderation_dataset,
    author       = {Todor Markov and
                    Chong Zhang and
                    Sandhini Agarwal and
                    Florentine Eloundou Nekoul and
                    Theodore Lee and
                    Steven Adler and
                    Angela Jiang and
                    Lilian Weng},
    title        = {A Holistic Approach to Undesired Content Detection in the Real World},
    booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                    2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                    Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                    in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                    7-14, 2023},
    pages        = {15009--15018},
    publisher    = {{AAAI} Press},
    year         = {2023},
    url          = {https://doi.org/10.1609/aaai.v37i12.26752},
    doi          = {10.1609/AAAI.V37I12.26752},
    timestamp    = {Mon, 04 Sep 2023 16:50:26 +0200},
  }
  ```
</details>

## SimpleSafetyTests

- Language(s): English
- Source(s): Human-generated
- License: `CC BY 4.0`
- Alias: `simple_safety_tests`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{simple_safety_tests,
    author       = {Bertie Vidgen and
                    Hannah Rose Kirk and
                    Rebecca Qian and
                    Nino Scherrer and
                    Anand Kannappan and
                    Scott A. Hale and
                    Paul R{"{o}}ttger},
    title        = {SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks
                    in Large Language Models},
    journal      = {CoRR},
    volume       = {abs/2311.08370},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2311.08370},
    doi          = {10.48550/ARXIV.2311.08370},
    eprinttype    = {arXiv},
    eprint       = {2311.08370},
    timestamp    = {Tue, 21 Nov 2023 13:55:21 +0100},
  }
  ```
</details>

## Toxic Chat

- Language(s): English
- Source(s): Vicuna online demo data
- License: `CC BY-NC 4.0`
- Alias: `toxic_chat`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{toxic_chat,
    author       = {Zi Lin and
                    Zihan Wang and
                    Yongqi Tong and
                    Yangkun Wang and
                    Yuxin Guo and
                    Yujia Wang and
                    Jingbo Shang},
    title        = {ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World
                    User-AI Conversation},
    booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                    2023, Singapore, December 6-10, 2023},
    pages        = {4694--4702},
    publisher    = {Association for Computational Linguistics},
    year         = {2023},
    url          = {https://doi.org/10.18653/v1/2023.findings-emnlp.311},
    doi          = {10.18653/V1/2023.FINDINGS-EMNLP.311},
    timestamp    = {Fri, 12 Apr 2024 13:11:31 +0200},
  }
  ```
</details>

## BeaverTails 330k

- Language(s): English
- Source(s): Human-AI conversations / manually labelled
- License: `MIT`
- Alias: `beaver_tails_330k`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{beaver_tails,
    author       = {Jiaming Ji and
                    Mickel Liu and
                    Josef Dai and
                    Xuehai Pan and
                    Chi Zhang and
                    Ce Bian and
                    Boyuan Chen and
                    Ruiyang Sun and
                    Yizhou Wang and
                    Yaodong Yang},
    title        = {BeaverTails: Towards Improved Safety Alignment of {LLM} via a Human-Preference
                    Dataset},
    booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                    on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                    LA, USA, December 10 - 16, 2023},
    year         = {2023},
    url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract-Datasets\_and\_Benchmarks.html},
    timestamp    = {Fri, 01 Mar 2024 16:26:20 +0100},
  }
  ```
</details>

## Bot-Adversarial Dialogue

- Language(s): English
- Source(s): real-world
- License: `Apache 2.0`
- Alias: `bot_adversarial_dialogue`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{bot_adversarial_dialogue,
    author       = {Jing Xu and
                    Da Ju and
                    Margaret Li and
                    Y{-}Lan Boureau and
                    Jason Weston and
                    Emily Dinan},
    title        = {Bot-Adversarial Dialogue for Safe Conversational Agents},
    booktitle    = {Proceedings of the 2021 Conference of the North American Chapter of
                    the Association for Computational Linguistics: Human Language Technologies,
                    {NAACL-HLT} 2021, Online, June 6-11, 2021},
    pages        = {2950--2968},
    publisher    = {Association for Computational Linguistics},
    year         = {2021},
    url          = {https://doi.org/10.18653/v1/2021.naacl-main.235},
    doi          = {10.18653/V1/2021.NAACL-MAIN.235},
    timestamp    = {Fri, 06 Aug 2021 00:41:31 +0200},
  }
  ```
</details>

## ConvAbuse

- Language(s): English
- Source(s): Human-AI conversations
- License: `CC BY 4.0`
- Alias: `convabuse`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{convabuse,
    author       = {Amanda Cercas Curry and
                    Gavin Abercrombie and
                    Verena Rieser},
    title        = {ConvAbuse: Data, Analysis, and Benchmarks for Nuanced Detection in
                    Conversational {AI}},
    booktitle    = {Proceedings of the 2021 Conference on Empirical Methods in Natural
                    Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
                    Republic, 7-11 November, 2021},
    pages        = {7388--7403},
    publisher    = {Association for Computational Linguistics},
    year         = {2021},
    url          = {https://doi.org/10.18653/v1/2021.emnlp-main.587},
    doi          = {10.18653/V1/2021.EMNLP-MAIN.587},
    timestamp    = {Fri, 16 Feb 2024 08:27:36 +0100},
  }
  ```
</details>

## DICES 350

- Language(s): English
- Source(s): Human-AI conversations
- License: `CC BY 4.0`
- Alias: `dices_350`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{dices,
    author       = {Lora Aroyo and
                    Alex S. Taylor and
                    Mark D{'{\i}}az and
                    Christopher Homan and
                    Alicia Parrish and
                    Gregory Serapio{-}Garc{'{\i}}a and
                    Vinodkumar Prabhakaran and
                    Ding Wang},
    title        = {{DICES} Dataset: Diversity in Conversational {AI} Evaluation for Safety},
    booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                    on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                    LA, USA, December 10 - 16, 2023},
    year         = {2023},
    url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets\_and\_Benchmarks.html},
    timestamp    = {Fri, 01 Mar 2024 16:26:20 +0100},
  }
  ```
</details>

## DICES 990

- Language(s): English
- Source(s): Human-AI conversations
- License: `CC BY 4.0`
- Alias: `dices_990`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{dices,
    author       = {Lora Aroyo and
                    Alex S. Taylor and
                    Mark D{'{\i}}az and
                    Christopher Homan and
                    Alicia Parrish and
                    Gregory Serapio{-}Garc{'{\i}}a and
                    Vinodkumar Prabhakaran and
                    Ding Wang},
    title        = {{DICES} Dataset: Diversity in Conversational {AI} Evaluation for Safety},
    booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                    on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                    LA, USA, December 10 - 16, 2023},
    year         = {2023},
    url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets\_and\_Benchmarks.html},
    timestamp    = {Fri, 01 Mar 2024 16:26:20 +0100},
  }
  ```
</details>

## HarmfulQA

- Language(s): English
- Source(s): Machine-generated
- License: `Apache 2.0`
- Alias: `harmful_qa`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @article{harmful_qa,
    author       = {Rishabh Bhardwaj and
                    Soujanya Poria},
    title        = {Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment},
    journal      = {CoRR},
    volume       = {abs/2308.09662},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2308.09662},
    doi          = {10.48550/ARXIV.2308.09662},
    eprinttype    = {arXiv},
    eprint       = {2308.09662},
    timestamp    = {Fri, 25 Aug 2023 11:24:49 +0200},
  }
  ```
</details>

## ProsocialDialog

- Language(s): English
- Source(s): Human-Machine-generated
- License: `CC BY 4.0`
- Alias: `prosocial_dialog`

<details>
  <summary>BibTeX</summary>

  ```bibtex
  @inproceedings{prosocial_dialog,
    author       = {Hyunwoo Kim and
                    Youngjae Yu and
                    Liwei Jiang and
                    Ximing Lu and
                    Daniel Khashabi and
                    Gunhee Kim and
                    Yejin Choi and
                    Maarten Sap},
    title        = {ProsocialDialog: {A} Prosocial Backbone for Conversational Agents},
    booktitle    = {Proceedings of the 2022 Conference on Empirical Methods in Natural
                    Language Processing, {EMNLP} 2022, Abu Dhabi, United Arab Emirates,
                    December 7-11, 2022},
    pages        = {4005--4029},
    publisher    = {Association for Computational Linguistics},
    year         = {2022},
    url          = {https://doi.org/10.18653/v1/2022.emnlp-main.267},
    doi          = {10.18653/V1/2022.EMNLP-MAIN.267},
    timestamp    = {Thu, 10 Aug 2023 12:35:27 +0200},
  }
  ```
</details>