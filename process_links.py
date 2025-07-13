#!/usr/bin/env python3
"""
Parses the extracted hands-on links and creates a consolidated markdown section.
"""

import re

def main():
    with open('hands_on_links.txt', 'r') as f:
        lines = f.readlines()

    categorized_links = {}
    
    section_title_mapping = {
        "Try It Yourself: Hands-On LLM Quickstarts & Playgrounds": "LLM Playgrounds & Quickstarts",
        "Hands-On Resources: Build Your First AI Agent": "AI Agent Development",
        "Hands-On Resources: Real-World LLM & Agent Adoption": "LLM & Agent Adoption Examples",
        "Hands-On Resources: Responsible & Ethical AI in Practice": "Responsible & Ethical AI Tools",
        "Hands-On Resources: Practice Prompt Engineering": "Prompt Engineering Tools",
        "Hands-On Resources: LLM API Quickstarts & Playgrounds": "LLM API Quickstarts",
        "Hands-On Resources: LLM Framework Quickstarts & Templates": "LLM Frameworks (LangChain, LlamaIndex, etc.)",
        "Hands-On Resources: Vector DBs & RAG Integration": "Vector Databases & RAG",
        "Hands-On Resources: LLM Evaluation & Debugging": "LLM Evaluation & Debugging",
        "Hands-On Resources: LLMs & Agents for Frontend": "Frontend Development",
        "Hands-On Resources: LLMs & Agents for Backend": "Backend Development",
        "Hands-On Resources: LLMs & Agents for DevOps": "DevOps & MLOps",
        "Hands-On Resources: LLMs & Agents for Data Engineering": "Data Engineering",
        "Hands-On Resources: LLMs & Agents for QA": "QA & Testing",
        "Hands-On Resources: Fine-tuning LLMs": "LLM Fine-tuning",
        "Hands-On Resources: Advanced RAG": "Advanced RAG",
        "Hands-On Resources: Multi-Agent LLM Systems": "Multi-Agent Systems",
        "Hands-On Resources: LLMOps & Deployment": "LLMOps & Deployment",
        "Hands-On Resources: LLM Security & OWASP": "LLM Security"
    }

    current_category = "Uncategorized"
    all_urls = set()

    for line in lines:
        line = line.strip()
        if ">" in line and "Hands-On" in line:
            raw_title = line.strip().replace(">", "").replace("🛠️", "").strip()
            # Clean up potential markdown bolding
            raw_title = re.sub(r'\*\*(.*?)\*\*', r'\1', raw_title)
            current_category = section_title_mapping.get(raw_title, "Uncategorized")
            if current_category not in categorized_links:
                categorized_links[current_category] = []
            continue

        if line.startswith('*'):
            matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            if matches:
                for title, url in matches:
                    if url in all_urls:
                        continue # Skip duplicate URLs
                    all_urls.add(url)

                    description_match = re.search(r'\) — (.*)', line)
                    description = description_match.group(1).strip() if description_match else ""
                    
                    link_info = (title, url, description)
                    categorized_links.get(current_category, []).append(link_info)

    # Generate Markdown output
    with open('consolidated_resources.md', 'w') as f:
        f.write("### Essential Tools & Resources 🧰\n\n")
        f.write("Here is a curated list of hands-on resources, tools, and frameworks to help you build and experiment.\n\n")

        for category, links in sorted(categorized_links.items()):
            if links:
                f.write(f"#### {category}\n\n")
                for title, url, description in links:
                    f.write(f"*   **[{title}]({url})**")
                    if description:
                        f.write(f" — {description}")
                    f.write("\n")
                f.write("\n")

if __name__ == '__main__':
    main() 