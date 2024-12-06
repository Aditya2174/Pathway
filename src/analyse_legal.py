from llama_index.core.llms import ChatMessage, MessageRole
import nltk
# Ensure NLTK is ready for sentence tokenization
nltk.download('punkt')


def split_document_by_sentences(document: str, sentences_per_chunk: int) -> list[str]:
    """
    Split a document into chunks of sentences for incremental analysis.
    """
    sentences = nltk.sent_tokenize(document)
    print(len(sentences))
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = sentences[i:i + sentences_per_chunk]
        # Add the last sentence of the previous chunk for continuity, if applicable
        if i > 0:
            chunk = sentences[i-10:i - 1] + chunk
        chunks.append(" ".join(chunk))
    return chunks

def analyze_document_with_context(gemini_model, document: str, user_query: str, sentences_per_chunk: int = 768) -> list[dict]:
    """
    Analyze a document incrementally with context preservation using the Gemini model.
    """
    results = []
    previous_response = ""  # Initialize the cumulative response
    response = ""

    chunks = split_document_by_sentences(document, sentences_per_chunk)

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}...")
        if idx != len(chunks)-1:
        # Prepare the prompt for Gemini
            system_prompt = (
                f"User Query: {user_query}\n"
                "Risk Analysis System: Update analysis based on current findings.\n\n"
                "### Scope Definition:\n"
                "- Decide analysis type: **Legal**, **Financial**, or **Combined**.\n\n"
                "### Key Areas of Analysis:\n\n"
                "#### Legal Risk Analysis:\n"
                "- **Contractual Obligations**: Are obligations clear and feasible? Risk of disputes or breaches?\n"
                "- **Warranties/Representations**: Accuracy and achievability; liability risks.\n"
                "- **Indemnification**: Clarity on responsibilities; fairness of terms.\n"
                "- **Limitation of Liability**: Adequacy of protection.\n"
                "- **Dispute Resolution**: Favorable methods (e.g., arbitration, litigation)?\n"
                "- **Governing Law**: Jurisdiction's laws and impact on outcomes?\n"
                "- **Force Majeure**: Realistic definitions and implications on non-performance?\n\n"
                "#### Financial Risk Analysis:\n"
                "- **Market Risk**: Impact of rate/price changes (e.g., interest, commodities)?\n"
                "- **Credit Risk**: Risks of delayed payments or defaults?\n"
                "- **Liquidity Risk**: Ability to meet short-term obligations?\n"
                "- **Operational Risk**: Vulnerabilities in processes, systems, or personnel?\n"
                "- **Legal/Regulatory Risk**: Compliance gaps and associated penalties?\n"
                "- **Strategic Risk**: Impacts of market dynamics, competition, or external events?\n\n"
                "### Instructions:\n"
                "1. **Scope Selection**: Determine whether to perform Legal, Financial, or Combined analysis.\n"
                "2. **Reconcile Findings**: Merge prior and current insights, resolving conflicts.\n"
                "3. **Highlight Updates**: Add new risks or mitigation strategies; retain relevant prior analysis.\n"
                "4. **Context Preservation**: Integrate seamlessly without losing previously analyzed points.\n"
                "5. **Concise Focus**: Exclude unrelated content while ensuring clarity.\n\n"
                "### Formatting Guidelines:\n"
                "- Separate findings into **Legal** and **Financial** sections if analyzing both.\n"
                "- Ensure continuity by merging relevant prior points with updated findings.\n"
                "- Avoid duplications; retain prior context without unnecessary rephrasing.\n\n"
                "### Analysis Template:\n"
                "- Update all relevant points based on the current section.\n"
                "- Use concise key points to retain clarity and avoid information loss.\n\n"
                "Previous Analysis:\n"
                f"{previous_response}\n\n"
                "Current Section:\n"
                f"{chunk}\n\n"
            )
        else:
            system_prompt = (
                "Risk Analysis System: Update the analysis of previous sections based on current findings.\n\n"
                "### Key Considerations:\n"
                "Before performing the analysis, determine the scope:\n"
                "- **Legal Analysis Only**: Focus exclusively on legal risks and obligations.\n"
                "- **Financial Analysis Only**: Concentrate solely on financial risks and vulnerabilities.\n"
                "- **Combined Analysis**: Address both legal and financial risks comprehensively.\n\n"
                "### Legal Risk Analysis Points:\n"
                "- **Contractual Obligations**: Are the obligations clear and feasible? Could they lead to disputes or breaches?\n"
                "- **Warranties and Representations**: Are they accurate and achievable? Could they expose you to liability?\n"
                "- **Indemnification**: Who is responsible for losses or damages? Are the indemnification provisions fair and balanced?\n"
                "- **Limitation of Liability**: Are the limits reasonable and sufficient to protect your interests?\n"
                "- **Dispute Resolution**: Is the chosen method (e.g., mediation, arbitration, litigation) favorable to your position?\n"
                "- **Governing Law**: Which jurisdiction's laws will apply? Could this impact dispute outcomes?\n"
                "- **Force Majeure**: Are events clearly and realistically defined? Could they excuse non-performance?\n\n"
                "### Financial Risk Analysis Points:\n"
                "- **Market Risk**: How could changes in market rates or prices (e.g., interest rates, commodity prices) affect operations?\n"
                "- **Credit Risk**: Are there risks of delayed payments or defaults by borrowers or customers?\n"
                "- **Liquidity Risk**: Can the organization meet short-term obligations with current cash flow and reserves?\n"
                "- **Operational Risk**: Are internal processes, systems, or people creating vulnerabilities?\n"
                "- **Legal and Regulatory Risk**: Are all compliance obligations met to avoid penalties and reputational harm?\n"
                "- **Strategic Risk**: Could market dynamics, competition, or external events impact long-term goals?\n\n"
                "### Instructions:\n"
                "1. **Scope Definition**: Decide whether to perform Legal, Financial, or Combined analysis based on the content of the current section.\n"
                "2. **Reconcile Findings**: Merge and update findings from previous responses with new insights from the current chunk.\n"
                "3. **Conflict Resolution**: Address any discrepancies between previous and current findings.\n"
                "4. **Add New Insights**: Highlight newly identified risks or mitigation strategies without losing previously provided information.\n"
                "5. **Preserve Context**: Retain and include previous analysis points that are still relevant but not explicitly present in the current chunk.\n"
                "6. **Avoid Extraneous Information**: Focus exclusively on analysis, avoiding unrelated content.\n\n"
                "### Formatting Guidelines:\n"
                "- Organize findings into clearly separated sections for Legal and Financial risks if both are analyzed.\n"
                "- Maintain coherence and continuity by ensuring no information from the previous analysis is lost.\n"
                "- Integrate previous relevant findings with current points without duplicating sections.\n\n"
                "### Analysis Template:\n"
                "- Combine all relevant points from the previous analysis with updates from the current section.\n"
                "- Do not shorten or rephrase prior findings unnecessarily; keep the context intact.\n"
                "- Present a structured report in a professional format.\n\n"
                "Previous Analysis:\n"
                f"{previous_response}\n\n"
                "Current Section:\n"
                f"{chunk}\n\n"
            )

        # Prepare message list for Gemini
        message_list = [
            ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
            ChatMessage(content=chunk, role=MessageRole.USER)
        ]

        # Perform chat with Gemini
        try:
            response = gemini_model.chat(message_list)
            previous_response = response  # Update previous response for the next iteration
            results.append({"chunk": chunk, "gemini_response": response})
        except Exception as e:
            print(f"Error analyzing chunk {idx + 1} with Gemini: {e}")
            results.append({"chunk": chunk, "gemini_response": None, "error": str(e)})

    return response.message.content