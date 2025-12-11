from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dustycam.utils.image_gen import get_client

# Define data structure using Pydantic
class ProjectPlan(BaseModel):
    budget_usd: int = Field(
        description="The estimated budget in dollars, strictly as an integer."
    )
    subjects: list[str] = Field(
        description="A list of specific subjects the user wants to recognize (e.g., 'Elk', 'Moose')."
    )
    scenes: list[str] = Field(
        description="A list of likely scenes or environments where these subjects will be found."
    )
    clarifying_questions: list[str] = Field(
        description="A list of 3 clarifying questions to ask the user to improve the model results."
    )

def run_oneshot_workflow(user_input: str):
    """
    Starts an interactive OneShot workflow session for creating a camera model.
    """
    # Initialize the client
    client = get_client()
    if not client: return

    # Context (could be retrieved or static definition of the assistant's role/goal)
    retrieved_context = (
        "We are creating a custom AI camera model. "
        "The goal is to produce a comprehensive project plan including budget, subjects, scenes, and clarifying questions. "
        "If no budget is specified by the user, assume a default of 1 dollar."
    )

    augmented_prompt = f"""
<context>
{retrieved_context}
</context>

<instruction>
Using the context above, answer the following user request and output a structured ProjectPlan:
{user_input}
</instruction>
"""
    
    print(f"Analyzing request: {user_input}...")
    
    try:
        # Initialize Chat Session
        chat = client.chats.create(
            model="gemini-2.0-flash-exp",
            config=types.GenerateContentConfig(
                system_instruction="You are an expert AI camera designer.",
                response_mime_type="application/json",
                response_schema=ProjectPlan,
            )
        )

        while True:
            # Send message (first time uses augmented_prompt, subsequent times uses user input directly/augmented)
            response = chat.send_message(augmented_prompt)

            # Access the structured data directly
            plan: ProjectPlan = response.parsed

            if not plan:
                print("Failed to parse project plan.")
                return

            # Output the results
            print(f"\n--- Project Plan Draft ---")
            print(f"💰 Budget: ${plan.budget_usd}")
            print(f"\n📸 Subjects to Track:")
            for subject in plan.subjects:
                print(f" - {subject}")
                
            print(f"\n🌲 Likely Scenes:")
            for scene in plan.scenes:
                print(f" - {scene}")

            print(f"\n❓ Questions for User:")
            for q in plan.clarifying_questions:
                print(f" - {q}")
            
            # Interactive Loop
            print("\n------------------------------------------------------------")
            print("Type your response to the questions above, or refining details.")
            print("Press ENTER to approve the plan and proceed.")
            print("Type 'exit' to quit.")
            
            next_input = input("> ").strip()
            
            if not next_input:
                print("Plan approved! Proceeding with data generation...(Not implemented yet)")
                break
            
            if next_input.lower() in ["exit", "quit"]:
                print("Exiting.")
                return

            print("\nUpdating plan based on your feedback...")
            # Update prompt for next iteration
            augmented_prompt = next_input

    except Exception as e:
        print(f"Error during project analysis: {e}")
