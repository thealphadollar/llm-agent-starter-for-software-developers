# LLMs and AI Agents: A Practical Learning Roadmap for Software Engineers (v2.0)

> *Last Updated: July 13, 2024*

A community-driven guide to help software engineers navigate the world of AI/LLMs, with a focus on practical application and hands-on learning.

## Table of Contents

* [Why AI for Engineers?](#why-ai-for-engineers-)
* [Quick Start: 30-Minute Challenge](#quick-start-30-minute-challenge-)
* [Common Knowledge: The AI Engineering Toolkit](#common-knowledge-the-ai-engineering-toolkit-%EF%B8%8F)
  * [Prompt Engineering](#prompt-engineering-%EF%B8%8F)
  * [Interacting with LLMs: APIs and SDKs](#interacting-with-llms-apis-and-sdks-)
  * [Frameworks and Libraries (e.g., LangChain, LlamaIndex)](#frameworks-and-libraries-eg-langchain-llamaindex-%EF%B8%8F)
  * [Vector Databases](#vector-databases-)
  * [System Insight: Observability, Evaluation & Reliability](#system-insight-observability-evaluation--reliability-)
  * [Operational Integrity: Security & Data Privacy](#operational-integrity-security--data-privacy-%EF%B8%8F)
  * [Resource Management: Cost & Performance](#resource-management-cost--performance-%EF%B8%8F)
* [Specialization: Backend Engineering](#specialization-backend-engineering-%EF%B8%8F)
* [Other Specializations (Community Contributions Welcome!)](other-specializations-community-contributions-welcome-%EF%B8%8F-%EF%B8%8F-%EF%B8%8F-)
  * [For Frontend Engineers](#for-frontend-engineers-%EF%B8%8F)
  * [For DevOps Engineers](#for-devops-engineers-%EF%B8%8F)
  * [For Data Engineers](#for-data-engineers-%EF%B8%8F)
  * [For QA Engineers](#for-qa-engineers-)
* [Advanced Topics (Optional Deep Dive)](#advanced-topics-optional-deep-dive-)
  * [Fine-tuning LLMs](#fine-tuning-llms-%EF%B8%8F)
  * [Retrieval Augmented Generation (RAG) - Deep Dive](#retrieval-augmented-generation-rag---deep-dive-)
  * [Multi-Agent Systems](#multi-agent-systems-)
  * [MLOps for LLMs (LLMOps)](#mlops-for-llms-llmops-%EF%B8%8F)
  * [Security for LLM Applications](#security-for-llm-applications-%EF%B8%8F)
* [Staying Updated & Community Engagement](#staying-updated--community-engagement-)
* [How to Contribute](#how-to-contribute-)
* [License](#license-)
* [Disclaimer](#disclaimer-)

## Why AI for Engineers? 🤔

The rise of powerful LLMs and AI Agents represents a fundamental shift in how we design, build, and maintain software. For software engineers, this isn't just another trend—it's a transformation of our tools, workflows, and capabilities. Understanding and adapting to these technologies is crucial for several key reasons:

* **🚀 Boost Productivity & Efficiency:** Automate repetitive coding tasks like generating boilerplate, writing unit tests, and debugging. Let AI agents handle complex workflows, freeing you to focus on high-level architecture and problem-solving.
* **💡 Unlock New Product Capabilities:** Build entirely new user experiences with sophisticated chatbots, personalized content, and AI-powered analytics. Integrate natural language interfaces into existing products to make them more intuitive and accessible.
* **📈 Evolve Your Skillset & Role:** Transition from writing every line of code to orchestrating, guiding, and validating the output of AI systems. Skills in prompt engineering, LLM APIs, and agentic frameworks are becoming essential.
* **🏆 Stay Competitive & Relevant:** Companies are rapidly adopting AI to gain a competitive edge. Proficiency in AI/LLM development makes you a high-demand asset in a fast-changing job market.
* **🧩 Solve More Complex Problems:** Tackle challenges that were previously too difficult for traditional software approaches, especially those involving unstructured data, nuanced understanding, or complex decision-making.

Embracing LLMs and AI agents isn't about replacing software engineers—it's about empowering us with a new class of tools to build more powerful, intelligent, and efficient software solutions than ever before.

<details>
<summary><b>Real-World Impact: How Developers are Winning with AI</b></summary>

Here are a few examples of how engineers are leveraging AI to transform their workflows and careers:

* **The AI-Augmented Developer: Boosting Daily Productivity**
  * **Finding:** Instead of a full role change, many developers are integrating AI tools directly into their existing workflows. In a large-scale study, developers using GitHub Copilot were found to complete tasks **55% faster** than those who didn't. This translates to automating repetitive work like writing boilerplate code, generating unit tests, and drafting documentation, freeing up significant time for complex problem-solving.
  * **Source:** [Read the full research on GitHub's Blog](https://github.blog/news-insights/research/research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness/)

* **From Software Engineer to AI Engineer: A Practical Journey**
  * **Finding:** The path into AI Engineering is often paved with practical, hands-on experience. As documented in *The Pragmatic Engineer*, AI Engineer Janvi Kalra made the transition from a traditional software role by dedicating her free time to building LLM-powered apps and participating in hackathons. After an initial rejection for an internal transfer, her new, self-taught expertise made her a key contributor at her company (Coda) and eventually led to a role at OpenAI.
  * **Source:** [Listen to Janvi Kalra's story on The Pragmatic Engineer Podcast](https://newsletter.pragmaticengineer.com/i/164588024/takeaways)

</details>

## Quick Start: 30-Minute Challenge 🚀

Ready to dive in and see the power of LLMs firsthand? This challenge will guide you through building a genuinely useful tool in under 30 minutes: **an AI-powered conventional commit message generator.**

**The Goal:** Create a command-line tool that reads your staged `git diff` and generates a concise, well-formatted commit message.

**Why this challenge?** It's a perfect first step:

* **Solves a Real Problem:** Automates a common, sometimes tedious, developer task.
* **Immediate "Wow" Factor:** Shows the practical power of LLMs with minimal code.
* **Hands-On Learning:** You'll make your first LLM API call and handle real-world data (a git diff).

---

### Step 1: Prerequisites (5 mins)

1. **Python:** Ensure you have Python 3.7+ installed.
2. **OpenAI Account & API Key:**
    * Sign up at [platform.openai.com](https://platform.openai.com/).
    * Navigate to the [API Keys section](https://platform.openai.com/api-keys) and create a new secret key. Copy it immediately—you won't be able to see it again.
3. **Set Environment Variable:** For security, don't hardcode your key. Open your terminal and set it as an environment variable.

    * **macOS/Linux:**

        ```bash
        export OPENAI_API_KEY='your-key-goes-here'
        ```

    * **Windows (Command Prompt):**

        ```bash
        set OPENAI_API_KEY=your-key-goes-here
        ```

    > **Note:** This variable is only set for your current terminal session. For a permanent solution, add it to your shell's profile file (e.g., `.zshrc`, `.bash_profile`).

4. **Install OpenAI Library:**

    ```bash
    pip install openai
    ```

### Step 2: The Code (15 mins)

Create a new file named `commit.py` and paste the following code into it.

```python
import os
import sys
from openai import OpenAI

# 1. Check for API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# 2. Read the git diff from standard input
try:
    diff_content = sys.stdin.read()
    if not diff_content:
        print("Error: No git diff provided. Pipe a diff to this script.")
        print("Example: git diff --staged | python commit.py")
        sys.exit(1)
except Exception as e:
    print(f"Error reading from stdin: {e}")
    sys.exit(1)

# 3. Define the prompt for the LLM
# This prompt guides the AI to generate a high-quality commit message.
system_prompt = """
You are an expert software developer who writes concise, high-quality conventional commit messages.
A conventional commit message has the following structure:
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

Based on the git diff provided, generate a conventional commit message.
- The commit message should be in the present tense.
- The description should be a short, imperative summary of the changes.
- The body should explain the 'why' behind the changes, not the 'what'.
- Only include a body if the changes are complex.
- Do not include the 'Signed-off-by' footer.
"""

# 4. Make the API Call
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini", # A fast and cost-effective model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the git diff:\n\n{diff_content}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    commit_message = response.choices[0].message.content.strip()
    
    # 5. Print the result
    print("--- Suggested Commit Message ---")
    print(commit_message)

except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    sys.exit(1)

```

### Step 3: Run Your Tool! (10 mins)

1. Open a terminal in a project that uses `git`.
2. Make a code change. For example, add a new function or fix a typo.
3. Stage your changes: `git add .`
4. Now, run your new tool by "piping" the `git diff` output into your Python script:

    ```bash
    git diff --staged | python commit.py
    ```

**Congratulations!** You should see an AI-generated conventional commit message printed in your terminal. You've just built your first practical LLM-powered developer tool.

---

### Next Steps: More Quick Wins

Feeling confident? Here are a couple more hands-on tutorials to build on what you've learned. They introduce new ideas like using external APIs and building in a different language.

* **Build an AI Stock Info Agent:** This tutorial from HackerNoon guides you through building a more advanced agent that can fetch real-time stock prices and company information using an external finance API. It's a great next step to learn how to give your AI tools to interact with the world.
  * **Source:** [Build Your First AI Agent on HackerNoon](https://hackernoon.com/ai-agents-for-beginners-building-your-first-ai-agent)

* **Create a Text Summarizer CLI in Node.js:** This tutorial from DEV Community shows you how to build a text summarizer tool, but this time using Node.js. It's a great way to see how the same core concepts apply in a different programming ecosystem.
  * **Source:** [Build an AI CLI Tool in Node.js on DEV.to](https://dev.to/mrflamez_/building-your-first-ai-cli-tool-using-openais-api-1d4a)

## Common Knowledge: The AI Engineering Toolkit 🛠️📖

This section covers the foundational concepts and cross-cutting concerns that every software engineer must understand. Mastering these will give you the ability to build, deploy, and manage robust, reliable, and efficient LLM-powered applications.

### Essential Tools & Resources 🧰

To get hands-on, you don't need dozens of tools at once. Focus on these essentials first. They provide a solid foundation for building and experimenting. For a more exhaustive list of tools for different specializations, see the [Comprehensive Resource Hub](#comprehensive-resource-hub-) at the end of this guide.

* **LLM Playgrounds (No Code Required):**
  * **[OpenAI Playground](https://platform.openai.com/playground):** Experiment interactively with GPT models.
  * **[Hugging Face LLM Spaces](https://huggingface.co/collections/hysts/llm-spaces-65250c035b29204d6d60d2bc):** Try out hundreds of open-source LLMs directly in your browser.
* **Core Development Frameworks:**
  * **[LangChain](https://python.langchain.com/):** The most popular open-source framework for building LLM applications. Start with their [Quickstart Guide](https://python.langchain.com/docs/get_started/quickstart).
  * **[LlamaIndex](https://www.llamaindex.ai/):** A framework specialized for connecting your private data to LLMs (the core of RAG). See their [10-line RAG example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/).
* **Vector Databases (for RAG & Memory):**
  * **[ChromaDB](https://www.trychroma.com/):** An open-source, developer-friendly embedding database perfect for getting started locally. Check out the [Quickstart](https://docs.trychroma.com/getting-started).
  * **[Pinecone](https://www.pinecone.io/):** A popular managed vector database for building scalable, production-ready RAG applications.
* **Evaluation & Debugging:**
  * **[LangSmith](https://www.langchain.com/langsmith):** An observability platform to trace, debug, and monitor your LLM applications. It's invaluable for understanding what's happening inside your chains and agents.

### Prompt Engineering ✍️💡

**Prompt engineering** is the art and science of crafting clear, effective inputs (prompts) to guide an LLM toward a desired output. It is the most fundamental skill for interacting with AI. Good prompting is the difference between a generic, unhelpful response and a nuanced, accurate one.

<details>
<summary><b>Key Prompting Techniques</b></summary>

Here are a few foundational techniques to get you started. As you progress, you'll find that combining these is key to solving complex problems.

* **Zero-Shot Prompting:** The simplest form. You ask the model to perform a task directly, without providing any prior examples. This relies on the model's pre-existing knowledge.
  * **Example:** `"Summarize this article."`

* **Few-Shot Prompting:** You provide a small number of examples (shots) of the task in the prompt. This helps the model understand the desired format, style, or logic.
  * **Example:** `"Translate English to French. sea otter -> loutre de mer. cheese -> fromage. car ->"`

* **Chain-of-Thought (CoT) Prompting:** You instruct the model to "think step-by-step" or "work out its reasoning" before giving the final answer. This dramatically improves performance on tasks requiring logical deduction or multi-step reasoning.
  * **Example:** `"Q: A juggler has 15 balls. He drops 5 and gives 2 to a friend. How many balls does he have left? A: Let's think step by step. The juggler starts with 15 balls. He drops 5, so he has 15 - 5 = 10 balls. Then he gives 2 away, so 10 - 2 = 8 balls. The final answer is 8."`

</details>

#### Key Resources

* **[Prompt Engineering Guide](https://www.promptingguide.ai/):** A comprehensive, interactive guide covering everything from basic techniques to advanced agentic prompting. An essential read.
* **[OpenAI's Prompt Engineering Cookbook](https://cookbook.openai.com/examples/gpt4-1_prompting_guide):** Practical examples, tips, and recipes from OpenAI for getting the most out of their models.
* **[LearnPrompting.org](https://learnprompting.org/):** A hands-on platform with structured tutorials and challenges to practice and test your skills.

### Interacting with LLMs: APIs and SDKs 🤝💻

Building applications requires programmatic access via **APIs** (Application Programming Interfaces) and **SDKs** (Software Development Kits). They are the bridge between your code and the AI's "brain," enabling you to integrate LLM capabilities into any service or script. The most common APIs are from **OpenAI**, **Anthropic**, **Google**, and **Hugging Face**.

### Frameworks and Libraries (e.g., LangChain, LlamaIndex) 📚🏗️

Frameworks accelerate development by providing reusable components and abstractions for common tasks like chaining multiple LLM calls, managing prompts, and connecting to data sources. They handle the boilerplate code, letting you focus on your application's high-level logic. Key players include **LangChain**, **LlamaIndex**, **CrewAI**, and **AutoGen**.

### Vector Databases 💾🔍

Vector databases provide a scalable, long-term memory for AI applications by storing and querying **embeddings**—numerical representations of data that capture semantic meaning. They are the core technology behind **Retrieval Augmented Generation (RAG)**, helping reduce hallucinations and improve factual accuracy by providing relevant context to the LLM.

For more information, read the Pinecone's [Vector Database](https://www.pinecone.io/learn/vector-database/) guide.

### System Insight: Observability, Evaluation & Reliability 📊📈

Building reliable LLM applications requires deep visibility into their complex, often non-deterministic behavior.

* **What it is:** A combination of practices for understanding the internal state of your LLM agents (observability), testing their output for quality and accuracy (evaluation), and ensuring they handle errors gracefully (reliability).
* **Why it's important:** When an LLM application fails, it can be hard to pinpoint the cause. A good observability and evaluation strategy is essential for debugging and building trust in your application.
* **Key Tools & Techniques:**
  * **Tracing:** The most critical technique. Use platforms like **LangSmith** to visualize the entire execution flow—every LLM call, tool input/output, and retrieved document.
  * **Evaluation:** Unlike traditional software, you can't just check if the output is `true` or `false`. You need to evaluate the *quality* of responses using automated metrics (e.g., **Ragas** for RAG pipelines) and human review.
  * **Logging & Error Handling:** Implement comprehensive logging for prompts and responses, along with robust error handling and fallback mechanisms.

### Operational Integrity: Security & Data Privacy 🛡️

Building with LLMs introduces unique security challenges. You are responsible for protecting against new attack vectors and handling data responsibly.

* **Why it's important:** A compromised LLM application can leak sensitive data, give malicious actors control over your tools, or generate harmful content.
* **Key Principles:**
  * **Treat LLM outputs as untrusted input:** Always sanitize and validate responses.
  * **Enforce least privilege:** Grant LLM-integrated tools and agents only the permissions they absolutely need.
  * **Protect user data:** Be mindful of data sent to third-party APIs and comply with privacy regulations.
  * **Consult the OWASP Top 10 for LLMs:** This is the industry-standard guide for mitigating LLM security risks.

### Resource Management: Cost & Performance 💰⚡️

LLM applications have direct operational costs and performance considerations that differ from traditional software.

* **Why it's important:** Unmanaged API usage can lead to surprise bills, while high latency creates a poor user experience. Balancing model capability, cost, and speed is a core engineering challenge.
* **Key Principles:**
  * **Monitor Token Usage:** Track API costs, which are typically based on input/output tokens.
  * **Choose the Right Model:** Use smaller, faster, cheaper models for simpler tasks.
  * **Optimize & Cache:** Use efficient prompts and cache results for repetitive queries to reduce token consumption and improve latency.

## Specialization: Backend Engineering ⚙️🧱

For backend engineers, LLMs and AI agents are not just tools—they are becoming a foundational part of the server-side stack. They enable the creation of highly intelligent services, automate complex business logic, and provide new ways to interact with data. This section outlines the key skills and provides hands-on tutorials to get you started.

### Key Use Cases

* **Automating Business Logic & Workflows:** Go beyond hard-coded rules and state machines. Use AI agents to orchestrate complex, multi-step business processes (e.g., user onboarding, fraud detection, order processing) that can adapt to real-time data and make nuanced decisions.
* **Building an Intelligent Data Layer:** Create services that can query databases using natural language. Build agents that can extract, transform, and validate data from unstructured sources like PDFs and emails, turning them into clean, structured data for your applications.
* **Smarter API Development & Management:** Automate the generation of boilerplate code, API documentation (like OpenAPI specs), and even entire CRUD endpoints. Agents can also power intelligent API gateways that transform requests or dynamically route traffic based on context.

### Hands-On Backend Projects

The best way to learn is by building. These tutorials will guide you through creating powerful, practical backend AI applications.

* **1. Build a RAG-Powered API with FastAPI**
  * **What you'll learn:** This is the quintessential backend AI task. You'll learn how to build a production-ready API that takes user queries, retrieves relevant documents from a vector store, and streams back answers from an LLM. It's a complete, end-to-end RAG implementation.
  * **Source:** [**Production-Ready RAG with FastAPI & LangChain** (Blog by Pradip Nichite)](https://blog.futuresmart.ai/building-a-production-ready-rag-chatbot-with-fastapi-and-langchain)

* **2. Create a Natural Language to SQL Agent**
  * **What you'll learn:** Unlock the data in your relational databases. This official LangChain tutorial teaches you how to build an agent that can translate human questions (e.g., "How many active users are in Germany?") into precise SQL queries, execute them, and return a natural language answer.
  * **Source:** [**Build a Q&A system over SQL data** (Official LangChain Docs)](https://python.langchain.com/docs/tutorials/sql_qa/)

* **3. Automate a Business Process with Multiple AI Agents**
  * **What you'll learn:** See how to orchestrate a team of AI agents to automate a complex workflow. This tutorial uses CrewAI to automate an entire data science project—from data collection and cleaning to model training and evaluation—showcasing the power of multi-agent collaboration.
  * **Source:** [**Data Science Automation with CrewAI** (Medium Article by Bhavik Jikadara)](https://medium.com/ai-agent-insider/data-science-automation-a-step-by-step-guide-using-crewai-e1468823e0f8)

## Other Specializations (Community Contributions Welcome!) 🖼️💻 🚀⚙️ 📊🛠️ 🧪🐞

While this guide provides a deep dive for backend engineering, the principles of AI engineering are universal. LLMs and agents are transforming every aspect of software development. The following sections are placeholders for community-driven learning paths.

**Want to contribute?** We are looking for experts to help build out these sections! Please see our [**Contribution Guidelines**](#how-to-contribute-) to learn how you can share your knowledge.

### For Frontend Engineers 🖼️💻

Frontend engineers can leverage LLMs to accelerate development by generating components, enhance user experiences with intelligent interfaces, and even automate accessibility improvements. This specialization focuses on integrating AI directly into the user-facing parts of an application.

> *This section is a work in progress. Community contributions are welcome!*

### For DevOps Engineers 🚀⚙️

For DevOps engineers, AI agents can automate complex CI/CD workflows, provide intelligent monitoring and incident response, and assist in managing Infrastructure as Code (IaC). This path is about using AI to improve the reliability and efficiency of development and operations.

#### Key Use Cases

* **Intelligent Incident Response & Self-Healing:** AI agents can monitor logs and metrics, perform root cause analysis on production alerts (e.g., Kubernetes pod crashes), and even execute automated remediation steps like restarting a service or rolling back a failed deployment.
* **Automated CI/CD Troubleshooting:** Reduce time-to-resolution for broken builds by using LLMs to analyze pipeline logs, explain cryptic error messages in plain English, and suggest concrete fixes for dependency conflicts or script failures.
* **Natural Language Infrastructure & Operations (ChatOps):** Interact with your systems using plain English. Build ChatOps bots (for Slack, MS Teams, etc.) that can generate IaC configurations (Terraform, Kubernetes YAML), query cloud resource status, or summarize system health on demand.

#### Hands-On Tutorials

* **[End-to-End Multi-Agent System with CI/CD](https://dev.to/aws-builders/end-to-end-testing-and-deployment-of-a-multi-agent-ai-system-with-docker-langgraph-and-circleci-4gee)** - A comprehensive tutorial on building, testing, and deploying a multi-agent AI system with Docker, LangGraph, and CircleCI, providing a full-stack, real-world example.
* **[Build a Real-Time Kubernetes AI Agent](https://www.digitalocean.com/community/tutorials/real-time-k8s-agent-genai-platform)** - This guide walks you through creating an AI-powered assistant that can generate and validate Kubernetes manifests, troubleshoot issues, and retrieve live cluster details.
* **[Create a DevOps ChatBot for Microsoft Teams](https://saimeda.medium.com/build-your-own-devops-copilot-with-mistral-7b-langchain-microsoft-teams-and-aws-5406daab9ecd)** - Learn how to build a DevOps copilot inside Microsoft Teams using an open-source LLM (Mistral 7B), LangChain, and AWS Lambda to interact with your infrastructure.

#### Key Considerations

* **Security & Permissions:** Giving an AI agent access to production infrastructure is high-risk. Always apply the principle of least privilege and require human approval for any state-changing actions (e.g., deployments, resource deletion).
* **Reliability & Human Oversight:** LLM outputs can be non-deterministic. For critical operations, implement a "human-in-the-loop" pattern where an engineer must approve the agent's proposed plan before execution.
* **Contextual Awareness:** For an agent to be effective, it needs deep context about your specific environment, runbooks, and toolchains. This often requires using RAG to ground the agent in your internal documentation.
* **Cost Management:** High-frequency operations, like real-time log analysis or constant system monitoring with a powerful LLM, can become expensive. Use smaller, faster models where possible and implement caching.

> *This section is a work in progress. Community contributions are welcome!*

### For Data Engineers 📊🛠️

Data engineers can use LLMs to build smarter data pipelines, automate data quality checks, and extract structured information from unstructured sources. This specialization covers how AI can enhance data ingestion, transformation, and management.

> *This section is a work in progress. Community contributions are welcome!*

### For QA Engineers 🧪🐞

QA engineers can employ AI to generate comprehensive test cases, create realistic test data, and even build self-healing test scripts. This specialization focuses on leveraging AI to achieve more intelligent, adaptive, and thorough quality assurance.

> *This section is a work in progress. Community contributions are welcome!*

## Advanced Topics (Optional Deep Dive) 🌌

This section is for those who want to go beyond the basics and understand the more complex techniques that power state-of-the-art LLM applications.

### Fine-tuning LLMs ⚙️🔧

**Fine-tuning** is the process of taking a pre-trained general-purpose LLM and further training it on a smaller, domain-specific dataset. This adapts the model to excel at a specific task, learn a particular style, or understand a niche knowledge domain.

* **When to Use Fine-tuning (vs. RAG):** While RAG provides external knowledge, fine-tuning is better for teaching the model a *skill* or a *style*. Use it when you need the model to learn a specific format, adopt a consistent persona, or understand a large body of knowledge that's difficult to fit into a prompt.
* **Parameter-Efficient Fine-Tuning (PEFT):** Full fine-tuning (updating all model weights) is computationally expensive. PEFT methods significantly lower this barrier by updating only a small subset of the model's parameters.
  * **LoRA (Low-Rank Adaptation):** Injects smaller, trainable matrices into the model's layers. This is the most popular PEFT technique.
  * **QLoRA (Quantized LoRA):** An even more efficient method that uses 4-bit quantization to reduce memory usage, making it possible to fine-tune very large models on a single GPU.

#### Key Resources

* **[A Definitive Guide to QLoRA](https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337):** A comprehensive guide with code examples for fine-tuning a 7B model using QLoRA.
* **[Hugging Face PEFT Library](https://huggingface.co/docs/peft/index):** The official documentation for the library that implements LoRA, QLoRA, and other PEFT methods.
* **[Fine-Tuning Open LLMs in 2025 with Hugging Face](https://blog.cubed.run/fine-tuning-open-llms-in-2025-with-hugging-face-c7ad75efabab):** A recent guide covering the latest techniques and best practices.

### Retrieval Augmented Generation (RAG) - Deep Dive 🧠🔗

Basic RAG retrieves relevant documents and passes them to an LLM. Advanced RAG optimizes every step of this process to deliver more accurate and contextually-aware answers.

* **Pre-Retrieval (Chunking & Embedding):**
  * **Chunking Strategy:** Go beyond fixed-size chunks. Use **semantic chunking** (splitting based on meaning) or **content-aware chunking** (e.g., splitting by paragraphs or sections) to preserve context.
  * **Embedding Optimization:** The quality of your embeddings is critical. Test different models from the **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** to find the best one for your data.
* **Retrieval (Fetching the Best Context):**
  * **Hybrid Search:** Combine traditional keyword search (like BM25) with semantic vector search to get the best of both worlds—precision and conceptual relevance.
  * **Query Transformations:** Use an LLM to improve the user's query *before* retrieval. Techniques like **[HyDE](https://arxiv.org/abs/2212.10496)** (generating a hypothetical document) or **Step-Back Prompting** (generating a more general query) can significantly improve retrieval accuracy.
  * **Re-ranking:** Retrieve a larger set of initial documents and then use a more sophisticated model (like a **[Cross-Encoder](https://www.sbert.net/docs/pretrained_models.html#cross-encoders)** or an LLM) to re-rank the results and select the most relevant ones.
* **Post-Retrieval (Processing Context):**
  * **Context Filtering & Compression:** Use an LLM to filter out irrelevant retrieved chunks or to summarize lengthy passages before sending them to the final generation step. This reduces noise and manages the context window.
  * **Iterative Refinement (e.g., [SELF-RAG](https://arxiv.org/abs/2310.11511)):** Create systems where the LLM can reflect on the retrieved documents and iteratively retrieve more information if the initial context is insufficient.

#### Key Resources

* **[Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6):** A comprehensive study of advanced RAG techniques and algorithms, with helpful diagrams.
* **[Improve RAG results in your LLM applications from basic to advanced](https://medium.com/@csakash03/improve-rag-results-in-your-llm-applications-from-basic-to-advanced-c3078356811f):** A practical guide covering strategies for pre-retrieval, retrieval, and post-retrieval.
* **[LangChain & LlamaIndex Docs:** Both frameworks have extensive documentation on advanced RAG pipelines, including code examples for all the techniques mentioned above.

### Multi-Agent Systems 🤖🤝🤖

**Multi-Agent Systems** involve multiple AI agents collaborating to solve complex problems that a single agent would struggle with. Each agent can have a specialized role, different tools, and distinct knowledge, working together towards a common objective.

* **Why use multiple agents?**
  * **Task Decomposition:** Break down complex problems into smaller, manageable sub-tasks for specialized agents (e.g., a "researcher" agent and a "writer" agent).
  * **Diverse Perspectives:** Combine agents with different skills or "personalities" to achieve more robust solutions (e.g., a "bull" and "bear" agent for financial analysis).
  * **Improved Reasoning:** Agents can debate, critique each other's work, and collectively arrive at a better solution.

* **Popular Frameworks:**
  * **[CrewAI](https://www.crewai.com/):** A framework designed to orchestrate role-playing, autonomous AI agents. It emphasizes collaborative intelligence and is great for hierarchical team structures.
  * **[AutoGen (Microsoft)](https://microsoft.github.io/autogen/):** An open-source framework that enables building applications with multiple agents that converse with each other to solve tasks. It's highly flexible and powerful for research and complex workflows.
  * **[LangGraph (LangChain)](https://langchain-ai.github.io/langgraph/):** Extends the LangChain ecosystem to build stateful, multi-agent applications by modeling them as graphs. It's excellent for creating cyclical and complex agentic behaviors.

#### Key Resources

* **[Comparative Analysis of LLM Agent Frameworks](https://medium.com/@josefsosa/white-paper-comparative-analysis-of-llm-agent-frameworks-3d9ea8c0212f):** A white paper comparing CrewAI, LangGraph, and AutoGen, with strengths and ideal use cases for each.
* **[Building Coordinated AI Agents with LangGraph: A Hands-On Tutorial](https://codecut.ai/building-multi-agent-ai-langgraph-tutorial/):** A practical tutorial for building a multi-agent investment committee with LangGraph.
* **[Introduction to Multi-Agent LLM Orchestration (GoPenAI Blog)](https://blog.gopenai.com/how-to-easily-build-an-llm-multi-agent-team-abcca8628d9e):** A tutorial on building a multi-agent app with a supervisor architecture using LangGraph.

### MLOps for LLMs (LLMOps) 🛠️🔄

**LLMOps** adapts traditional Machine Learning Operations (MLOps) for the unique challenges of managing the lifecycle of Large Language Models. It provides the practices and tools to reliably deploy, monitor, and maintain LLM-based applications in production.

* **What Makes LLMOps Different?**
  * **Prompt Engineering & Management:** Prompts are a critical part of the application and must be versioned, tested, and optimized like code.
  * **Focus on Inference Cost & Latency:** Managing token usage and API latency is a primary operational concern.
  * **Specialized Evaluation:** Traditional ML metrics (like accuracy) are often insufficient. LLMOps requires metrics for fluency, coherence, factual consistency (in RAG), and safety.
  * **Vector Database Management:** For RAG systems, the vector database is a core production component that needs to be managed and updated.

* **The LLMOps Lifecycle:**
    1. **Data Management:** Collect, clean, and version data for fine-tuning or RAG.
    2. **Prompt Engineering:** Design, test, and version control your prompts.
    3. **Experiment Tracking:** Use tools like **Weights & Biases** or **MLflow** to log experiments with different models, prompts, and hyperparameters.
    4. **Evaluation:** Rigorously test your application using both automated metrics and human-in-the-loop review.
    5. **Deployment:** Serve the LLM application using scalable infrastructure (e.g., Kubernetes, serverless functions).
    6. **Monitoring:** Use tools like **LangSmith** or **Arize** to track performance, cost, latency, and data drift in real-time.
    7. **Continuous Improvement:** Use feedback from monitoring to iteratively improve your prompts, data, or models.

#### Key Resources

* **[Mastering LLMOps: Deploy, Manage, and Scale LLMs on AWS, GCP, and Azure](https://medium.com/@williamwarley/mastering-llmops-deploy-manage-and-scale-large-language-models-on-aws-gcp-and-azure-0ef073c7274d):** A comprehensive guide with step-by-step examples for deploying LLMs on major cloud platforms.
* **[What is LLMOps? (LinkedIn Article)](https://www.linkedin.com/pulse/what-llmops-tips-implement-best-practices-nitin-kumar-sharma-ergnc):** A clear overview of LLMOps, its core components, and how it differs from traditional MLOps.
* **[The LLMOps Lifecycle (ML Contant)](https://www.mlcontant.com/llmops):** An in-depth look at the stages of the LLMOps lifecycle with recommended tools for each stage.

### Security for LLM Applications 🛡️

Securing LLM applications involves addressing a new class of vulnerabilities that go beyond traditional web application security. The **OWASP Top 10 for Large Language Model Applications** is the industry-standard guide for understanding and mitigating these risks.

* **The Core Risks:**
  * **`LLM01: Prompt Injection`**: The most critical risk. Attackers craft malicious inputs to manipulate the LLM's behavior, bypass safety instructions, or cause unintended actions. This can be direct (user input) or indirect (hidden instructions in a document the LLM processes).
  * **`LLM02: Sensitive Information Disclosure`**: The LLM unintentionally reveals sensitive data (PII, financial records, trade secrets) that was present in its training data or provided in a user's session.
  * **`LLM04: Data and Model Poisoning`**: Attackers manipulate training data or RAG sources to introduce biases, vulnerabilities, or backdoors into the model.
  * **`LLM05: Improper Output Handling`**: The application blindly trusts the LLM's output without sanitizing it, leading to downstream vulnerabilities like XSS, SQL injection, or command injection.
  * **`LLM06: Excessive Agency`**: The LLM is granted too much autonomy to interact with other systems, allowing attackers to trick it into performing unauthorized actions (e.g., making purchases, deleting files).
  * **`LLM08: Vector and Embedding Weaknesses`**: Specific to RAG systems, where attackers can poison vector databases or craft queries to retrieve sensitive information.

#### Key Resources

* **[OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/):** The official source. Every developer building with LLMs should be familiar with this list.
* **[Breaking Down OWASP Top 10 LLM (2025)](https://medium.com/@appsecwarrior/breaking-down-owasp-top-10-llm-2025-cd99ed46761b):** A detailed analysis of each risk with exploitation techniques, mitigation strategies, and test cases.
* **[OWASP Top 10 for LLM Applications & Generative AI: Key Updates for 2025](https://www.lasso.security/blog/owasp-top-10-for-llm-applications-generative-ai-key-updates-for-2025):** A summary of what's new in the latest version of the list.

## Staying Updated & Community Engagement 🌐🤝

The field of AI is evolving at an unprecedented pace. Staying updated and engaging with the community are crucial for any software engineer in this domain.

### Key Newsletters & Blogs 📰✍️

* **The Batch (DeepLearning.AI):** Andrew Ng's weekly newsletter covering important AI news and breakthroughs.
* **Import AI (Jack Clark):** A highly-regarded weekly newsletter with detailed analysis of cutting-edge AI research.
* **Ben's Bites:** A popular daily AI newsletter with concise summaries of everything happening in AI.
* **Hugging Face Blog:** Updates on new models, datasets, and libraries from a leading AI community.
* **OpenAI Blog:** Official announcements and research from one of the leading AI companies.

### Research Papers & Pre-print Servers 📄🔬

* **[arXiv.org](https://arxiv.org/corr/) (cs.AI, cs.CL, cs.LG sections):** The primary pre-print server for AI research. Most significant papers appear here first.
* **[Papers with Code](https://paperswithcode.com/):** An invaluable resource that links research papers to their code implementations.

### Online Communities 🗣️💻

* **Reddit:**
  * **[r/LargeLanguageModels](https://www.reddit.com/r/LargeLanguageModels/):** Discussions on new models, research, and applications.
  * **[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/):** Focused on running LLMs locally on your own hardware.
* **Discord Servers:** Many open-source projects (LangChain, LlamaIndex, CrewAI, etc.) have their own Discord servers for community support and discussion.
* **X (Formerly Twitter):** Follow key researchers, developers, and companies in the AI/LLM space for real-time updates.

### Contributing to Open Source 🧑‍💻🤝

Contributing to open-source LLM projects is one of the best ways to learn.

* **Find a Project:** Look for projects you use, like **[LangChain](https://github.com/langchain-ai/langchain)**, **[LlamaIndex](https://github.com/run-llama/llama_index)**, or libraries on **[Hugging Face](https://github.com/huggingface)**.
* **Start Small:** Look for `good first issue` labels. Fixing a typo in the documentation is a valuable contribution!

## Comprehensive Resource Hub 📚

This section contains a categorized list of hands-on resources, tools, and frameworks. Use this as a reference as you dive deeper into specific areas of interest.

<details>
<summary><b>Click to expand the full list of resources.</b></summary>

#### AI Agent Development

* **[LangChain Agents Quickstart (Python Docs)](https://python.langchain.com/docs/tutorials/agents/)** — Step-by-step guide to building your first agent with LangChain.
* **[CrewAI Quickstart (Official Docs)](https://docs.crewai.com/getting-started/quickstart)** — Build collaborative, role-based agents with CrewAI.
* **[AutoGen Basic Tutorial (Microsoft)](https://microsoft.github.io/autogen/0.2/docs/tutorial/introduction/)** — Create multi-agent conversations and workflows with AutoGen.
* **[LangChain Agent Tutorial Notebooks (GitHub)](https://github.com/langchain-ai/langchain/tree/master/docs/docs/tutorials)** — Community-contributed agent demos.

#### Backend Development

* **[LangChain Agents for Backend (Docs)](https://python.langchain.com/docs/tutorials/agents/)** — Build tool-using agents for backend workflows.
* **[CrewAI Backend Agent Example (GitHub)](https://github.com/joaomdmoura/crewai-examples/tree/main/backend)** — Multi-agent backend orchestration demo.
* **[Text-to-SQL with LLMs (LangChain Docs)](https://python.langchain.com/docs/tutorials/sql_qa/)** — Tutorial for building a natural language to SQL agent.
* **[Vanna AI (GitHub)](https://github.com/vanna-ai/vanna)** — Open-source natural language to SQL agent for databases.
* **[LLM-Powered Report Generation (Medium)](https://medium.com/@jagadeesan.ganesh/mastering-llm-ai-agents-building-and-using-ai-agents-in-python-with-real-world-use-cases-c578eb640e35)** — Guide to using agents for backend automation and reporting.

#### Data Engineering

* **[DEnGPT: Autonomous Data Engineer Agent (Substack)](https://juhache.substack.com/p/dengpt-autonomous-data-engineer-agent)** — Walkthrough of an agent building a data pipeline (Lambda, S3, Serverless Framework).
* **[RAG for Data Engineering (LlamaIndex Docs)](https://docs.llamaindex.ai/en/stable/examples/workflow/rag/)** — Example of using RAG for data extraction and enrichment.
* **[AI Agents for Data Engineering (Matillion Blog)](https://www.matillion.com/blog/ai-agents-data-engineering)** — Blog on agents for ETL, schema inference, and pipeline monitoring.
* **[Building LLM Applications With Vector Databases (Neptune.ai)](https://neptune.ai/blog/building-llm-applications-with-vector-databases)** — Guide to vectorizing and indexing data for semantic search.
* **[LlamaIndex Data Connectors (GitHub)](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/connectors)** — Community and official data loader templates.

#### DevOps & MLOps

* **[HolmesGPT (GitHub)](https://github.com/robusta-dev/holmesgpt)** — AI agent for investigating Kubernetes alerts, fetching logs, and correlating metrics.
* **[llm-opstower (GitHub)](https://github.com/opstower-ai/llm-opstower)** — CLI tool to query AWS services, CloudWatch metrics, and billing using LLMs.
* **[k8s-langchain (GitHub)](https://github.com/jjoneson/k8s-langchain)** — Agent to interact with Kubernetes clusters using LLMs.
* **[How AI Agents Will Transform DevOps Workflows (The New Stack)](https://thenewstack.io/how-ai-agents-will-transform-devops-workflows-for-engineers/)** — Blog on LLMs for IaC, monitoring, and more.
* **[Generative AI's Impact on Developers (DevOps.com)](https://devops.com/generative-ais-impact-on-developers/)** — AI agents for vulnerability scanning and patching.

#### Frontend Development

* **[ReactAgent.io (GitHub)](https://github.com/reactagentio/reactagent)** — Autonomous agent that generates React components from user stories.
* **[Building an AI agent for your frontend project (LogRocket Blog)](https://blog.logrocket.com/building-ai-agent-frontend-project/)** — Step-by-step guide to integrating LLMs in frontend apps.
* **[AI-Powered Search in React (Vercel Blog)](https://vercel.com/blog/7-ai-features-you-can-add-to-your-app-today)** — Guide to adding semantic search with LLMs to a Next.js app.
* **[How AI Agents Are Quietly Transforming Frontend Development (The New Stack)](https://thenewstack.io/how-ai-agents-are-quietly-transforming-frontend-development/)** — Blog on agent-driven UI refactoring and accessibility.

#### LLM & Agent Adoption Examples

* **[How GitHub Copilot Boosts Developer Productivity (GitHub Blog)](https://github.blog/2023-03-22-github-copilot-x-the-ai-powered-developer-experience/)** — Real-world impact of LLMs in software engineering.
* **[Generative AI’s Impact on Developers (DevOps.com)](https://devops.com/generative-ais-impact-on-developers/)** — Practical examples of GenAI in the SDLC.
* **[Awesome LLM Applications (GitHub)](https://github.com/hwchase17/awesome-llm-applications)** — Curated list of real-world LLM/agent-powered projects.

#### LLM API Quickstarts

* **[OpenAI API Quickstart (Python)](https://platform.openai.com/docs/quickstart?context=python)** — Official quickstart for using GPT models via API.
* **[Claude API Quickstart (Docs)](https://docs.anthropic.com/claude/docs/quickstart-guide)** — Get started with Claude models.
* **[Gemini API Quickstart (Python)](https://ai.google.dev/tutorials/python_quickstart)** — Step-by-step guide for Gemini models.
* **[Cohere API Quickstart (Docs)](https://docs.cohere.com/)** — Start using Cohere's Command models.
* **[Inference API Quickstart (Docs)](https://huggingface.co/docs/api-inference/quicktour)** — Run inference on thousands of models via API.

#### LLM Evaluation & Debugging

* **[LangSmith Quickstart (Docs)](https://docs.smith.langchain.com/observability/quick_start)** — Trace, debug, and evaluate LLM chains and agents.
* **[DeepEval Quickstart (GitHub)](https://github.com/confident-ai/deepeval#quickstart)** — Open-source framework for LLM evaluation with metrics and pytest integration.
* **[Ragas Quickstart (Docs)](https://docs.ragas.io/en/latest/getstarted/rag_eval/)** — Evaluate RAG pipelines with specialized metrics.
* **[W&B LLM Evaluation Guide (Docs)](https://wandb.ai/site/solutions/llmops)** — Track, compare, and visualize LLM experiments.

#### LLM Fine-tuning

* **[Fine-tune a Transformer Model (Hugging Face Course)](https://huggingface.co/course/chapter3/3?fw=pt)** — Step-by-step guide for fine-tuning on your own data.
* **[PEFT Library Docs](https://huggingface.co/docs/peft/index)** — Parameter-efficient fine-tuning (LoRA, QLoRA, Adapters).
* **[LoRA: Low-Rank Adaptation (arXiv)](https://arxiv.org/abs/2106.09685)** — Original paper.
* **[QLoRA: Efficient Finetuning (arXiv)](https://arxiv.org/abs/2305.14314)** — QLoRA method.

#### LLM Frameworks (LangChain, LlamaIndex, etc.)

* **[LangChain Getting Started (Python Docs)](https://python.langchain.com/docs/get_started/quickstart)** — Official quickstart for building LLM apps.
* **[LlamaIndex Quickstart (Docs)](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)** — Step-by-step guide for RAG and data-augmented LLM apps.
* **[CrewAI Quickstart (Docs)](https://docs.crewai.com/getting-started/quickstart)** — Build collaborative, role-based agents.
* **[AutoGen Getting Started (Microsoft Docs)](https://microsoft.github.io/autogen/0.2/docs/tutorial/introduction/)** — Multi-agent orchestration quickstart.

#### LLM Playgrounds & Quickstarts

* **[OpenAI Playground (official)](https://platform.openai.com/playground)** — Experiment interactively with GPT-4, GPT-3.5, and more.
* **[LLM Spaces Collection](https://huggingface.co/collections/hysts/llm-spaces-65250c035b29204d6d60d2bc)** — Try open-source LLMs (Llama, Mistral, Falcon, etc.) in your browser, no setup required.
* **[Google AI Studio Quickstart](https://ai.google.dev/gemini-api/docs/ai-studio-quickstart)** — Use Gemini models in a web playground.
* **[Unified Free LLM API Gateway: OpenRouter Guide (Hugging Face Blog)](https://huggingface.co/blog/lynn-mikami/llm-free)** — Access many top LLMs for free via a single API.
* **[3Blue1Brown: Large Language Models Explained (2024)](https://www.3blue1brown.com/lessons/mini-llm)** — Visual, intuitive intro to LLMs.

#### LLM Security

* **[OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)** — Official list of LLM-specific security risks and mitigations.
* **[Prompt Injection Attacks & Defenses (OWASP)](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)** — Learn to test and defend against prompt injection.
* **[Secure LLM App Patterns (genai.owasp.org)](https://genai.owasp.org/)** — Secure design patterns and checklists for LLM applications.

#### Multi-Agent Systems

* **[AutoGen Quickstart (Microsoft)](https://microsoft.github.io/autogen/0.2/docs/tutorial/introduction/)** — Build multi-agent LLM workflows.
* **[CrewAI Multi-Agent Example (Docs)](https://docs.crewai.com/getting-started/quickstart)** — Role-based agent collaboration.
* **[LangGraph Quickstart (Docs)](https://python.langchain.com/docs/langgraph/)** — Build graph-based multi-agent systems.

#### Prompt Engineering Tools

* **[Prompt Engineering Guide (promptingguide.ai)](https://www.promptingguide.ai/)** — Comprehensive, interactive guide with techniques, examples, and a playground.
* **[PromptPerfect Playground](https://promptperfect.jina.ai/)** — Optimize and test prompts interactively.
* **[Prompt Engineering Challenges (LearnPrompting)](https://learnprompting.org/docs/basics/prompting)** — Practice and test your skills with real-world prompt challenges.

#### QA & Testing

* **[Building AI Agents to Automate Software Test Case Creation (NVIDIA Blog)](https://developer.nvidia.com/blog/building-ai-agents-to-automate-software-test-case-creation/)** — Framework and code for LLM-driven test generation.
* **[LLM Agent Workflows for Full-stack Testing (Coforge Blog)](https://www.coforge.com/what-we-know/blog/using-llm-agent-workflows-for-improving-automating-deploying-a-reliable-full-stack-web-application-testing-process)** — Multi-agent workflow for E2E, API, and security testing.
* **[A Complete Guide to AI Testing Agents for Software Testing (Kobiton)](https://kobiton.com/ai-agents-software-testing-guide/)** — Overview and practical tips for AI-powered test automation.
* **[Synthetic Test Data with LLMs (Medium)](https://medium.com/@petrbrzek/llm-for-test-data-generation-7e7e7e7e7e7e)** — Tutorial for generating diverse test data using LLMs.

#### Responsible & Ethical AI Tools

* **[Responsible AI Dashboard (Microsoft)](https://github.com/microsoft/responsible-ai-toolbox)** — Visualize, diagnose, and mitigate model fairness, explainability, and error analysis issues.
* **[Google Responsible AI Practices Checklist](https://ai.google/responsibilities/responsible-ai-practices/)** — Practical checklist for building responsible AI systems.
* **[Fairness Indicators (TensorFlow)](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide)** — Tool for evaluating model fairness and bias in ML workflows.
* **[Partnership on AI - Responsible Practices](https://partnershiponai.org/paper/responsible-publication-recommendations/)** — Resources and tools for ethical AI development.
* **[AI Fairness 360 (GitHub)](https://github.com/Trusted-AI/AIF360)** — Open-source toolkit to help detect and mitigate bias in machine learning models.

#### Vector Databases & RAG

* **[Pinecone Quickstart (Docs)](https://docs.pinecone.io/docs/quickstart)** — Step-by-step guide to creating and querying a vector DB.
* **[Weaviate Quickstart (Docs)](https://weaviate.io/developers/weaviate/quickstart)** — Launch and use Weaviate locally or in the cloud.
* **[Chroma Quickstart (Docs)](https://docs.trychroma.com/getting-started)** — Build a local vector DB in Python.
* **[Milvus Quickstart (Docs)](https://milvus.io/docs/quickstart.md)** — Deploy and use Milvus for vector search.
* **[Qdrant Quickstart (Docs)](https://qdrant.tech/documentation/quick-start/)** — Set up and query Qdrant.

</details>

## How to Contribute 🤝📝

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License 📜

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Disclaimer 📢

*Placeholder for disclaimer. The information is provided "as is" without any warranties.*
