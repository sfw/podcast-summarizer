import gradio as gr
from gradio_modal import Modal

class PromptRefiner:
    """
    A reusable class that provides iterative prompt-refinement logic with an OpenAI-like client.
    We store the conversation in self.conversation and produce Chatbot (user, AI) pairs.
    """

    def __init__(self, openai_client, system_instructions=None, model="o1-mini", temperature=1):
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.conversation = []
        
        if not system_instructions:
            system_instructions = (
                "You are an AI that helps refine the user's prompt. "
                "Ask clarifying questions or suggest improvements until the user is satisfied."
                "If you want any further information, please ask for it."
            )

        # Start with a system message
        self.conversation.append({"role": "developer", "content": system_instructions})

    def _call_ai(self, user_message=None):
        """
        Appends an optional user message, calls the AI, appends AI response, returns AI text.
        """
        if user_message:
            self.conversation.append({"role": "user", "content": user_message})
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature
        )
        ai_text = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": ai_text})
        return ai_text

    def start_refinement(self, prompt_text):
        """
        Clears old conversation (except system message) and starts a new round with the user's prompt.
        Returns the chat in Chatbot pairs.
        """
        system_msg = self.conversation[0]["content"]
        self.conversation = [{"role": "user", "content": system_msg}]

        user_intro = (
            f"I have a prompt I want to refine:\n\n{prompt_text}\n\n"
            "Please ask clarifying questions or suggest improvements. "
            "We'll iterate until I'm satisfied, then produce a final refined prompt."
        )
        self._call_ai(user_intro)
        return self._to_chatbot_pairs()

    def continue_refinement(self, user_message):
        """
        The user sends a new message. Returns updated Chatbot pairs.
        """
        self._call_ai(user_message)
        return self._to_chatbot_pairs()

    def finalize_refinement(self):
        """
        Tells the AI we're done and requests the final refined prompt.
        Returns that prompt text.
        """
        final_req = (
            "I am done refining. Please provide the final refined prompt"
            "and only the final prompt. If no feedback or refinements were given,"
            "please just send back the original prompt."
        )
        reply = self._call_ai(final_req)
        return self._extract_refined_prompt(reply)

#
# Setup to add future refinements if required
#
    def _extract_refined_prompt(self, ai_text):
        """
        Returns the final prompt.
        """
        return ai_text

    def _to_chatbot_pairs(self):
        """
        Converts self.conversation to a list of (user_text, ai_text) for gr.Chatbot display.
        System messages are skipped.
        """
        pairs = []
        pending_user = None
        for msg in self.conversation:
            if msg["role"] == "developer":
                continue
            elif msg["role"] == "user":
                pending_user = msg["content"]
            elif msg["role"] == "assistant":
                if pending_user is None:
                    pairs.append(("", msg["content"]))
                else:
                    pairs.append((pending_user, msg["content"]))
                pending_user = None
        return pairs


def add_prompt_refinement_modal(
    blocks: gr.Blocks,
    openai_client,
    main_prompt_textbox: gr.Textbox,
    system_instructions=None,
    model="o1-mini",
    temperature=1,
    refine_button_label="Refine Prompt"
):
    """
    Encapsulates the entire 'Refine Prompt' sidebar chatbot flow in a library function.
    - blocks: the main gr.Blocks() container
    - openai_client: your openai-like client (with .chat.completions.create)
    - main_prompt_textbox: the textbox whose content we want to refine & overwrite with the final refined prompt
    - system_instructions: optional system message for the refiner
    - model, temperature: model hyperparams
    - refine_button_label: label for the button that starts refinement

    Returns:
      1) The 'Refine Prompt' button in the main area
      2) A dictionary of relevant components if you need them (e.g. for further customization)
    """



    # Create the PromptRefiner instance
    refiner = PromptRefiner(
        openai_client=openai_client,
        system_instructions=system_instructions,
        model=model,
        temperature=temperature
    )

    with blocks.Row():
        refine_btn = gr.Button(refine_button_label)

    # The Modal refinement UI
    with Modal(visible=False, allow_user_close=True) as modal:
        gr.Markdown("### Prompt Refinement Chat")
        refine_chat_container = gr.Column()
        with refine_chat_container:
            chatbot = gr.Chatbot(label="Refinement Chat")
            user_input = gr.Textbox(label="Your Message", lines=3)
            send_btn = gr.Button("Send")
            done_btn = gr.Button("Done Refining")

    # 1) When "Refine Prompt" is clicked, we start the refinement
    def on_refine_click(prompt_text):
        chat_pairs = refiner.start_refinement(prompt_text)
        return chat_pairs

    refine_btn.click(
        fn=on_refine_click,
        inputs=main_prompt_textbox,
        outputs=[chatbot]
    )

    #show modal
    refine_btn.click(lambda: Modal(visible=True), None, modal)

    # 2) Sending user messages
    def on_user_send(user_msg, chat_state):
        new_pairs = refiner.continue_refinement(user_msg)
        return "", new_pairs  # clear user input, update chat

    send_btn.click(
        fn=on_user_send,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

    # 3) Done refining
    def on_done_refining(chat_state):
        final_prompt = refiner.finalize_refinement()
        # hide chat, overwrite main prompt
        return final_prompt

    done_btn.click(
        fn=on_done_refining,
        inputs=chatbot,
        outputs=main_prompt_textbox
    )

    done_btn.click(lambda: Modal(visible=False), None, modal)

    return refine_btn, {
        "chat_container": refine_chat_container,
        "chatbot": chatbot,
        "send_btn": send_btn,
        "done_btn": done_btn,
        "refiner_instance": refiner
    }