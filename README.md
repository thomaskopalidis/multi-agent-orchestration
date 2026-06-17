# Multi AI Agent Systems with CrewAI

Hands-on implementations of multi-agent AI systems built with [CrewAI](https://github.com/joaomdmoura/crewAI), based on the DeepLearning.AI course [*Multi AI Agent Systems with crewAI*](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai). Each notebook is a self-contained example demonstrating a different multi-agent orchestration pattern, from sequential content pipelines to collaborative agent crews with custom tools.

## What's inside

Each notebook defines a crew of role-playing agents (with a `role`, `goal`, and `backstory`), assigns them tasks with a clear `description` and `expected_output`, and orchestrates execution through CrewAI's `Crew` object.

| Notebook | Use case | Key concepts |
|---|---|---|
| `L2_research_write_article.ipynb` | Research & write a blog article | Sequential process, agent collaboration (Planner → Writer → Editor) |
| `L3_customer_support_automation.ipynb` | Automate customer support responses | Agent memory, guardrails, quality-focused tasks |
| `L4_tools_customer_outreach.ipynb` | Customer outreach campaign | Custom & built-in tools (search, scraping), tool delegation |
| `L5_automate_event_planning.ipynb` | Event planning automation | Structured/typed outputs, multiple coordinated agents |
| `L6_financial_analysis_collaboration.ipynb` | Financial analysis | Multi-agent collaboration, delegation between agents |
| `L7_tailor_job_applications.ipynb` | Tailor a resume/application to a job posting | Combining tools, tasks, and agents into an applied pipeline |

> Adjust the filenames above if your repo uses different names — these follow the course's lesson numbering (L2–L7).

## Example: Research & Write Article crew

The flagship example builds a three-agent content pipeline:

- **Content Planner** — researches the topic and produces an outline, audience analysis, and SEO keywords.
- **Content Writer** — turns the plan into a full blog post, grounded in the planner's research.
- **Editor** — proofreads the draft for tone, balance, and journalistic best practices.

```python
from crewai import Agent, Task, Crew

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic}...",
    allow_delegation=False,
    verbose=True
)

# writer and editor agents follow the same pattern...

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
```

Tasks run **sequentially** by default — each agent's output becomes context for the next, mirroring a real editorial workflow.

## Tech stack

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration framework
- [crewai_tools](https://github.com/joaomdmoura/crewAI-tools) — pre-built tools for search, scraping, and file I/O
- OpenAI API (`gpt-3.5-turbo`) as the underlying LLM
- LangChain Community integrations (optional alternative LLM backends, e.g. HuggingFace Hub, Mistral)

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
   ```

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export OPENAI_MODEL_NAME="gpt-3.5-turbo"
   ```

4. Launch Jupyter and run any notebook:
   ```bash
   jupyter notebook
   ```

## Notes

- Some notebooks use custom tools that may require additional API keys (e.g. a search API) — check the top of each notebook for specifics.
- Agents can be swapped to run on other LLMs (HuggingFace Hub, Mistral, etc.) — examples are included at the bottom of the relevant notebooks.

## Acknowledgments

Built while completing [*Multi AI Agent Systems with crewAI*](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai) by DeepLearning.AI and crewAI. All exercises adapted and run independently for this repository.
