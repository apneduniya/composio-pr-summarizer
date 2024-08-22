from composio_crewai import ComposioToolSet, Action
import json
from utils.github_patch import get_patch
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
import os
import dotenv
import typing as t


dotenv.load_dotenv()  # load env variables

REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

llm = ChatOpenAI(model="gpt-4o")  # Initialize the language model
composio_toolset = ComposioToolSet()

# Fetch all pull requests
action_response = composio_toolset.execute_action(
    action=Action.GITHUB_LIST_PULL_REQUESTS, params={"owner": REPO_OWNER, "repo": REPO_NAME})

# Initialize the summarizer agent
summarizer_agent = Agent(
    role="Github Pull Request Summarizer Agent",
    goal="""A detailed summary of the pull request including the code changes, and all the other details of the pull request.""",
    backstory="""You are an AI agent responsible for summarizing pull requests. 
        You need to summarize pull requests. Make sure to summarize all the details of the pull request.""",
    verbose=True,
    llm=llm,
    cache=False,
)

tasks: t.List[Task] = []

if not action_response["successfull"]:
    print("Error fetching pull request:", action_response["error"])

pr_list = action_response["data"]["details"]

if not pr_list:
    print("No pull requests found")


for pr in pr_list:
    title = pr["title"]
    body = pr["body"]
    assignees = pr["assignees"]
    labels = pr["labels"]
    patch = get_patch(pr["patch_url"])

    task = Task(
        description=f"""Summarize pull request and give a summary of the code change and other details in the pull request.
        Pull request title: {pr["title"]}
        Pull request body: {pr["body"]}
        Pull request patch: {patch}
        Pull request assignees: {assignees}
        Pull request labels: {labels}""",
        agent=summarizer_agent,
        expected_output="A summary of the pull request including code changes, files changed, author, assignees, and labels",
        async_execution=True
    )
    
    tasks.append(task)

    break

crew = Crew(
    agents=[summarizer_agent],
    tasks=tasks,
    verbose=True,
)

result = crew.kickoff()

print(result)


