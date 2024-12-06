from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, MessageRole
import nltk
import time
# Ensure NLTK is ready for sentence tokenization
nltk.download('punkt')

# Initialize Gemini model
google_api_key = "AIzaSyArUJfr-TuiJXXyN6hvVFWahl_TAO7t_g0"
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)

def split_document_by_sentences(document: str, sentences_per_chunk: int) -> list[str]:
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

def analyze_document_with_context(document: str, user_query: str, sentences_per_chunk: int = 768) -> list[dict]:
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

if __name__ == 'main':
    document = '''
    Free website disclaimer: cover
    1.	This template legal document was produced and published by Docular Limited.
    2.	We control the copyright in this template, and you may only use this template in accordance with the licensing provisions in our terms and conditions. Those licensing provisions include an obligation to retain the attribution / credit incorporated into the template.
    3.	You will need to edit this template before use. Guidance notes to help you do so are set out at the end of the template. During the editing process, you should delete those guidance notes and this cover sheet. Square brackets in the body of the document indicate areas that require editorial attention. "ORs" in the body of the document indicate alternative provisions. By the end of the editing process, there should be no square brackets left in the body of the document, and only one alternative from each set of alternatives should remain. Elements may be specified as optional in the accompanying notes, but that does not mean that they are in all cases removable. Depending upon the circumstances, an optional element may be: (i) required by law; or (ii) necessary to ensure that the document is internally consistent.
    4.	If you have any doubts about the editing or use of this template, you should seek professional legal advice.
    5.	You can request a quote for legal services (including the adaptation or review of a legal document produced from this template) using this form: https://docular.net/pages/contact.

    Website disclaimer
    1.	Introduction
    1.1	This disclaimer shall govern your use of our website.
    1.2	By using our website, you accept this disclaimer in full; accordingly, if you disagree with this disclaimer or any part of this disclaimer, you must not use our website.
    2.	Credit
    2.1	This document was created using a template from Docular (https://seqlegal.com/free-legal-documents/website-disclaimer).
    You must retain the above credit. Use of this document without the credit is an infringement of copyright. However, you can purchase from us an equivalent document that does not include the credit.
    3.	Copyright notice
    3.1	Copyright (c) [year(s) of first publication] [full name].
    3.2	Subject to the express provisions of this disclaimer:
    (a)	we, together with our licensors, own and control all the copyright and other intellectual property rights in our website and the material on our website; and
    (b)	all the copyright and other intellectual property rights in our website and the material on our website are reserved.
    4.	Permission to use website
    4.1	You may:
    (a)	view pages from our website in a web browser;
    (b)	download pages from our website for caching in a web browser; and
    (c)	print pages from our website[ for your own personal and non-commercial use][, providing that such printing is not systematic or excessive],
    [additional list items]
        subject to the other provisions of this disclaimer.
    4.2	Except as expressly permitted by Section 4.1 or the other provisions of this disclaimer, you must not download any material from our website or save any such material to your computer.
    4.3	You may only use our website for [[your own personal and business purposes]] OR [[define purposes]]; you must not use our website for any other purposes.
    4.4	Unless you own or control the relevant rights in the material, you must not:
    (a)	republish material from our website (including republication on another website);
    (b)	sell, rent or sub-license material from our website;
    (c)	show any material from our website in public;
    (d)	exploit material from our website for a commercial purpose; or
    (e)	redistribute material from our website.
    4.5	We reserve the right to suspend or restrict access to our website, to areas of our website and/or to functionality upon our website. We may, for example, suspend access to the website [during server maintenance or when we update the website]. You must not circumvent or bypass, or attempt to circumvent or bypass, any access restriction measures on the website.
    5.	Misuse of website
    5.1	You must not:
    (a)	use our website in any way or take any action that causes, or may cause, damage to the website or impairment of the performance, availability, accessibility, integrity or security of the website;
    (b)	use our website in any way that is unlawful, illegal, fraudulent or harmful, or in connection with any unlawful, illegal, fraudulent or harmful purpose or activity;
    (c)	hack or otherwise tamper with our website;
    (d)	probe, scan or test the vulnerability of our website without our permission;
    (e)	circumvent any authentication or security systems or processes on or relating to our website;
    (f)	use our website to copy, store, host, transmit, send, use, publish or distribute any material which consists of (or is linked to) any spyware, computer virus, Trojan horse, worm, keystroke logger, rootkit or other malicious computer software;
    (g)	[impose an unreasonably large load on our website resources (including bandwidth, storage capacity and processing capacity)];
    (h)	[decrypt or decipher any communications sent by or to our website without our permission];
    (i)	[conduct any systematic or automated data collection activities (including without limitation scraping, data mining, data extraction and data harvesting) on or in relation to our website without our express written consent];
    (j)	[access or otherwise interact with our website using any robot, spider or other automated means[, except for the purpose of [search engine indexing]]];
    (k)	[use our website except by means of our public interfaces];
    (l)	[violate the directives set out in the robots.txt file for our website];
    (m)	[use data collected from our website for any direct marketing activity (including without limitation email marketing, SMS marketing, telemarketing and direct mailing)]; or
    (n)	[do anything that interferes with the normal use of our website].
    [additional list items]
    5.2	You must not use data collected from our website to contact individuals, companies or other persons or entities.
    5.3	You must ensure that all the information you supply to us through our website, or in relation to our website, is [true, accurate, current, complete and non-misleading].
    6.	Limited warranties
    6.1	We do not warrant or represent:
    (a)	the completeness or accuracy of the information published on our website;
    (b)	that the material on the website is up to date;
    (c)	that the website will operate without fault; or
    (d)	that the website or any service on the website will remain available.
    [additional list items]
    6.2	We reserve the right to discontinue or alter any or all of our website services, and to stop publishing our website, at any time in our sole discretion without notice or explanation; and save to the extent expressly provided otherwise in this disclaimer, you will not be entitled to any compensation or other payment upon the discontinuance or alteration of any website services, or if we stop publishing the website.
    6.3	To the maximum extent permitted by applicable law and subject to Section 7.1, we exclude all representations and warranties relating to the subject matter of this disclaimer, our website and the use of our website.
    7.	Limitations and exclusions of liability
    7.1	Nothing in this disclaimer will:
    (a)	limit or exclude any liability for death or personal injury resulting from negligence;
    (b)	limit or exclude any liability for fraud or fraudulent misrepresentation;
    (c)	limit any liabilities in any way that is not permitted under applicable law; or
    (d)	exclude any liabilities that may not be excluded under applicable law.
    7.2	The limitations and exclusions of liability set out in this Section 7 and elsewhere in this disclaimer: 
    (a)	are subject to Section 7.1; and
    (b)	govern all liabilities arising under this disclaimer or relating to the subject matter of this disclaimer, including liabilities arising in contract, in tort (including negligence) and for breach of statutory duty, except to the extent expressly provided otherwise in this disclaimer.
    7.3	To the extent that our website and the information and services on our website are provided free of charge, we will not be liable for any loss or damage of any nature.
    7.4	We will not be liable to you in respect of any losses arising out of any event or events beyond our reasonable control.
    7.5	We will not be liable to you in respect of any business losses, including (without limitation) loss of or damage to profits, income, revenue, use, production, anticipated savings, business, contracts, commercial opportunities or goodwill.
    7.6	We will not be liable to you in respect of any loss or corruption of any data, database or software.
    7.7	We will not be liable to you in respect of any special, indirect or consequential loss or damage.
    8.	Variation
    8.1	We may revise this disclaimer from time to time.
    8.2	The revised disclaimer shall apply to the use of our website from the time of publication of the revised disclaimer on the website. 
    9.	Severability
    9.1	If a provision of this disclaimer is determined by any court or other competent authority to be unlawful and/or unenforceable, the other provisions will continue in effect.
    9.2	If any unlawful and/or unenforceable provision of this disclaimer would be lawful or enforceable if part of it were deleted, that part will be deemed to be deleted, and the rest of the provision will continue in effect. 
    10.	Law and jurisdiction
    10.1	This disclaimer shall be governed by and construed in accordance with [English law].
    10.2	Any disputes relating to this disclaimer shall be subject to the [exclusive] OR [non-exclusive] jurisdiction of the courts of [England].
    11.	Statutory and regulatory disclosures
    11.1	We are registered in [trade register]; you can find the online version of the register at [URL], and our registration number is [number].
    11.2	We are subject to [authorisation scheme], which is supervised by [supervisory authority].
    11.3	We are registered as [title] with [professional body] in [the United Kingdom] and are subject to [rules], which can be found at [URL].
    11.4	We subscribe to [code(s) of conduct], which can be consulted electronically at [URL(s)].
    11.5	Our VAT number is [number].
    12.	Our details
    12.1	This website is owned and operated by [name].
    12.2	We are registered in [England and Wales] under registration number [number], and our registered office is at [address].
    12.3	Our principal place of business is at [address].
    12.4	You can contact us:
    (a)	[by post, to [the postal address given above]];
    (b)	[using our website contact form];
    (c)	[by telephone, on [the contact number published on our website]]; or
    (d)	[by email, using [the email address published on our website]].
    [additional list items]

    Free website disclaimer: drafting notes
    This website disclaimer does three different jobs for a website operator: it disclaims certain liabilities that might arise out of the use of the website; it sets out the basis upon which a website may be used; and it prompts certain disclosures that are or may be required of website operators by English/EU law.
    The document is a shortened and simplified version of our standard website terms and conditions.
    If the website in respect of which the document will be used includes any ecommerce features, user-generated content features or other service-like features, you should consider one of our more detailed terms and conditions documents, rather than this disclaimer.
    Section 1: Introduction
    Section 1.2
    Optional element.
    The completed document should be easily accessible on the website, with a link from every page.
    Section 2: Credit
    Section: Free documents licensing warning
    Optional element. Although you need to retain the credit, you should remove the inline copyright warning from this document before use.
    Section 3: Copyright notice
    A copyright notice is an assertion of ownership.
    Copyright notices usually take the form specified in Article 3(1) of the Universal Copyright Convention (although the UCC itself is now of very limited significance):
    "Any Contracting State which, under its domestic law, requires as a condition of copyright, compliance with formalities such as deposit, registration, notice, notarial certificates, payment of fees or manufacture or publication in that Contracting State, shall regard these requirements as satisfied with respect to all works protected in accordance with this Convention and first published outside its territory and the author of which is not one of its nationals, if from the time of the first publication all the copies of the work published with the authority of the author or other copyright proprietor bear the symbol © accompanied by the name of the copyright proprietor and the year of first publication placed in such manner and location as to give reasonable notice of claim of copyright."
    It will be rare for a website owner to be the sole proprietor of all the copyright in a website. For example, the software code used to run the website may belong to another person. For this reason, the notice here refers also to licensors.
    •	Universal Copyright Convention - http://portal.unesco.org/en/ev.php-URL_ID=15381&URL_DO=DO_TOPIC&URL_SECTION=201.html
    •	Berne Convention for the Protection of Literary and Artistic Works - https://wipolex.wipo.int/en/text/283698
    Section 3.1
    •	What was the year of first publication of the relevant copyright material (or the range of years)?
    •	Who is the principal owner of copyright in the website?
    Section 4: Permission to use website
    Every website is a compendium of copyright-protected works. These may include literary works, (website text, HTML, CSS and software code), graphic works (photographs and illustrations), databases, sound recordings and films.
    The most fundamental principle of copyright law is that a person may not copy a protected work without permission. Using a website involves copying some or all of the works comprised in the website. Accordingly, a user needs permission to use a website. A "licence" is just such a permission.
    In most if not all cases, by publishing a website a person will be granting an implied licence to website visitors to copy of the website. The problem with an implied licence is that the scope of the licence is inherently uncertain. Is the visitor permitted to download the entire website? Is the visitor permitted to reproduce elements of the website elsewhere?
    Because of this uncertainty, most publishers will include an express licence setting out exactly what visitors are permitted to do in relation to a website and, just as important, what they are not permitted to do.
    The scope of the licence will vary. In editing these provisions, consider carefully exactly what your users should be allowed to do with the website and material on the website.
    •	Copyright, Designs and Patents Act 1988 - https://www.legislation.gov.uk/ukpga/1988/48
    Section 4.3
    Optional element.
    •	For what purposes may the website be used?
    Section 5: Misuse of website
    Section 5.1
    •	Should automated interactions with the website be prohibited?
    •	Will the website incorporate a robots.txt file?
    •	Should users be prohibited from using the website for direct marketing activity?
    Section 5.2
    Optional element. Should the use of data collected from the website to contact people and businesses be prohibited?
    Section 5.3
    Optional element.
    •	What standard of veracity etc should user-submitted content meet?
    Section 6: Limited warranties
    Section 6.1
    Optional element.
    Section 6.2
    Optional element.
    Section 7: Limitations and exclusions of liability
    Limitations and exclusions of liability are regulated and controlled by law, and the courts may rule that particular limitations and exclusions of liability are unenforceable.
    The courts may be more likely to rule that provisions excluding liability, as opposed to those merely limiting liability, are unenforceable.
    If there is a risk that any particular limitation or exclusion of liability will be found to be unenforceable by the courts, that provision should be drafted as an independent term, and be numbered separately from the other provisions. 
    It may improve the chances of a limitation or exclusion of liability being found to be enforceable if it was specifically drawn to the attention of the relevant person.
    In English law, exclusions and limitations of liability in legal notices are regulated by the Unfair Contract Terms Act 1977 ("UCTA"). 
    Legal notices regulated by UCTA cannot exclude or restrict a party's liability for death or personal injury resulting from negligence (Section 2(1), UCTA).
    Except insofar as the relevant term satisfies the requirements of reasonableness, such legal notices cannot exclude or restrict liability for negligence (Section 2(2), UCTA).
    These guidance notes provide a very incomplete and basic overview of a complex subject. Accordingly, you should take legal advice if you may wish to rely upon a limitation or exclusion of liability.
    •	Unfair Contract Terms Act 1977 - https://www.legislation.gov.uk/ukpga/1977/50
    Section 7.1
    Do not delete this provision (except upon legal advice). Without this provision, the specific limitations and exclusions of liability in the document are more likely to be unenforceable.
    Section 7.3
    Optional element. Do you want to attempt to exclude all liability for free services and information?
    This sort of exclusion is quite common, but unlikely to be enforceable in court.
    Section 7.5
    Optional element.
    Section 7.6
    Optional element.
    Section 7.7
    Optional element.
    Section 8: Variation
    Changes to legal documents published on a website will not generally be retrospectively effective, and variations without notice to and/or consent from relevant users may be ineffective.
    Section 10: Law and jurisdiction
    The questions of which law governs a document and where disputes relating to the document may be litigated are two distinct questions.
    Section 10.1
    This document has been drafted to comply with English law, and the governing law provision should not be changed without obtaining expert advice from a lawyer qualified in the appropriate jurisdiction. In some circumstances the courts will apply provisions of their local law, such as local competition law or consumer protection law, irrespective of a choice of law clause.
    •	Which law should govern the document?
    Section 10.2
    In some circumstances your jurisdiction clause may be overridden by the courts.
    •	Should the jurisdiction granted be exclusive or non-exclusive? Choose "non-exclusive" jurisdiction if you may want to enforce the terms and conditions against users outside England and Wales. Otherwise, choose "exclusive jurisdiction".
    •	The courts of which country or jurisdiction should adjudicate disputes under the document?
    Section 11: Statutory and regulatory disclosures
    Do the Electronic Commerce (EC Directive) Regulations 2002 apply to the website or is the website operator registered for VAT?
    This section can be deleted where website operator is not registered for VAT and the Electronic Commerce (EC Directive) Regulations 2002 do not apply. Generally, those Regulations will apply unless a website is entirely non-commercial, ie where a website does not offer any goods or services and does not involve any remuneration (which includes remuneration for carrying AdSense or other advertising).
    •	Electronic Commerce (EC Directive) Regulations 2002 (original version) - https://www.legislation.gov.uk/uksi/2002/2013/made
    Section 11.1
    Optional element. Is the website operator registered in a trade or similar register that is available to the public?
    The Electronic Commerce (EC Directive) Regulations 2002 provide that if you are "registered in a trade or similar register available to the public", you must provide "details of the register in which the service provider is entered and his registration number, or equivalent means of identification in that register".
    •	What is the name of the trade register?
    •	At what URL can the trade register be found?
    •	What is the website operator's registration number?
    •	Regulation 6, Electronic Commerce (EC Directive) Regulations 2002 - http://www.legislation.gov.uk/uksi/2002/2013/regulation/6/made
    Section 11.2
    Optional element. Is the website operator subject to an authorisation scheme (eg under financial services legislation)?
    The Electronic Commerce (EC Directive) Regulations 2002 provide that "where the provision of the service is subject to an authorisation scheme" you must provide "the particulars of the relevant supervisory authority".
    •	What is the name of the authorisation scheme to which the website operator is subject?
    •	What authority supervises the authorisation scheme?
    •	Regulation 6, Electronic Commerce (EC Directive) Regulations 2002 - http://www.legislation.gov.uk/uksi/2002/2013/regulation/6/made
    Section 11.3
    Optional element. Is the service provider a member of a regulated profession (eg solicitors)?
    The Electronic Commerce (EC Directive) Regulations 2002 provide that if "the service provider exercises a regulated profession", it must provide "(i) the details of any professional body or similar institution with which the service provider is registered; (ii) his professional title and the member State where that title has been granted; (iii) a reference to the professional rules applicable to the service provider in the member State of establishment and the means to access them".
    •	What is the website operator's professional title?
    •	Which professional body regulates the website operator?
    •	What is the name of the document containing the rules governing the profession?
    •	At what URL can the rules be found?
    •	Regulation 6, Electronic Commerce (EC Directive) Regulations 2002 - http://www.legislation.gov.uk/uksi/2002/2013/regulation/6/made
    Section 11.4
    Optional element. Does the website operator subscribe to any codes of conduct?
    The Electronic Commerce (EC Directive) Regulations 2002 provide that "a service provider shall indicate which relevant codes of conduct he subscribes to and give information on how those codes can be consulted electronically".
    •	Identify the codes of conduct in question.
    •	Where can the codes be viewed?
    •	Regulation 9, Electronic Commerce (EC Directive) Regulations 2002 - http://www.legislation.gov.uk/uksi/2002/2013/regulation/9/made
    Section 11.5
    Optional element. Is the website operator registered for VAT?
    •	What is the website operator's VAT number?
    Section 12: Our details
    Optional element.
    The provisions here reflect a mixture of EU law and UK law requirements relating to contact information.
    All services covered by the Ecommerce Directive (which was implemented in the UK through the Electronic Commerce (EC Directive) Regulations 2002) must provide a name, a geographic address (not a P.O. Box number) and an email address.
    Under distinct UK legislation, UK companies must provide their corporate names, their registration numbers, their place of registration and their registered office address on their websites (although not necessarily in this document). Sole traders and partnerships that carry on a business in the UK under a "business name" (i.e. a name which is not the name of the trader/names of the partners or certain other specified classes of name) must also make certain additional disclosures: (a) in the case of a sole trader, the individual's name; (b) in the case of a partnership, the name of each member of the partnership; and (c) in either case, in relation to each person named, an address in the UK at which service of any document relating in any way to the business will be effective. All operators covered by the Provision of Services Regulations 2009 must also provide a telephone number.
    •	Electronic Commerce (EC Directive) Regulations 2002 (original version) - https://www.legislation.gov.uk/uksi/2002/2013/made
    •	Provision of Services Regulations 2009 - https://www.legislation.gov.uk/uksi/2009/2999
    •	Directive 2000/31/EC (Directive on electronic commerce) - https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A32000L0031
    Section 12.1
    •	What is the name of the company, partnership, individual or other legal person or entity that owns and operates the website?
    Section 12.2
    Optional element. Is the relevant person a company?
    •	In what jurisdiction is the company registered?
    •	What is the company's registration number or equivalent?
    •	Where is the company's registered address?
    Section 12.3
    Optional element.
    •	Where is the relevant person's head office or principal place of business?
    Section 12.4
    Optional element.
    •	By what means may the relevant person be contacted?
    •	Where is the relevant person's postal address published?
    •	Either specify a telephone number or give details of where the relevant number may be found.
    •	Either specify an email address or give details of where the relevant email address may be found.

    '''
    start = time.time()
    results = analyze_document_with_context(document)
    end = time.time()
    print(results)
    print(end-start)