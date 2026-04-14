# LLMPrior

## Preparation

1. Install requirements: `pip install -r requirements.txt`

2. Create an OpenAI API key [here](https://platform.openai.com/api-keys).

3. Set system environment variable "OPENAI_API_KEY".

## Start

1. Download dataset and set `DATASET_ROOT` in `prompt.py`
2. Set `APP`, `REPEAT`, and `MODE` in `main.py`
3. Run `main.py`

### Code Structure

- **llm.py**: This file contains the interface for calling OpenAI's large language model service. 
- **prompt.py**: This file deals with prompt templates and prompt construction. It loads the report content from the test report files and constructs prompts accordingly.
- **parse.py**: This file is responsible for parsing the responses from the large language model. It parses the hierarchical clustering results (multi-branch tree) from the model's response based on predefined data structures, and then uses a iterative selection algorithm (function `generate_sequence`) to generate a priority prioritization sequence based on the parsing results. (**Note**: The robustness of the parsing process is not strong, and it may fail if the model does not output in the format specified in the prompt.)
- **main.py**: This is the entry point of the program. `run_query` is responsible for constructing prompts, calling the large language model, and obtaining responses; `run_parse` is responsible for parsing the responses and generating the priority prioritization results; `run_evaluate` is responsible for evaluating the prioritization results.

## Tips

- If `run_parse` fails, try manually fixing the non-formatting response content of LLM, or rerun the specific cases using `given_range`.

- If `run_evaluate` fails, the generated prioritization result is invalid. Rerun the specific cases using `given_range`.
