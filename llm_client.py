import os
import json
import base64
import logging
from typing import List, Optional

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None


from schemas import LLMClassificationResponse, PredictedClass
import config

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """
    Unified client supporting Google Gemini, OpenAI and Azure OpenAI.
    Provides structured text classification and image reading.
    (Embeddings for RAG are handled separately in vector_store.py with LangChain + Gemini.)
    All chat calls use temperature 0 (plus a fixed seed for OpenAI / Azure) for repeatable output.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        # Google Gemini parameters
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        # Direct OpenAI parameters
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        # Azure OpenAI parameters
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_chat_deployment: Optional[str] = None,
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.is_mock = False
        self._sampling_params = {"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED}

        if self.provider == "gemini":
            api_key = gemini_api_key or config.GEMINI_API_KEY
            self.chat_model = gemini_model or config.GEMINI_MODEL
            self.vision_model = self.chat_model  # Gemini models read images natively
            # The 'seed' parameter is not sent to Gemini's OpenAI-compatible endpoint; temperature 0 only
            self._sampling_params = {"temperature": config.LLM_TEMPERATURE}

            if self.is_placeholder_key(api_key):
                logger.warning(
                    "[LLMClient] Gemini API key not set or placeholder detected. "
                    "Running in simulated Mock Mode for demonstration."
                )
                self.is_mock = True
                self.client = None
            else:
                # Gemini exposes an OpenAI-compatible API (chat, structured output, images)
                self.client = OpenAI(api_key=api_key, base_url=config.GEMINI_BASE_URL)
                logger.info(f"[LLMClient] Connected to Google Gemini with model '{self.chat_model}'")

        elif self.provider == "azure":
            api_key = azure_api_key or config.AZURE_OPENAI_API_KEY
            endpoint = azure_endpoint or config.AZURE_OPENAI_ENDPOINT
            api_version = azure_api_version or config.AZURE_OPENAI_API_VERSION
            self.chat_model = azure_chat_deployment or config.AZURE_OPENAI_CHAT_DEPLOYMENT
            self.vision_model = self.chat_model

            if not api_key or "your-" in api_key or not endpoint or "your-" in endpoint:
                logger.warning(
                    "[LLMClient] Azure OpenAI credentials not set or placeholder detected. "
                    "Running in simulated Mock Mode for demonstration."
                )
                self.is_mock = True
                self.client = None
            else:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                )
                logger.info(f"[LLMClient] Connected to Azure OpenAI at {endpoint} (deployment: {self.chat_model})")

        elif self.provider == "openai":
            api_key = openai_api_key or config.OPENAI_API_KEY
            self.chat_model = openai_model or config.OPENAI_MODEL
            self.vision_model = self.chat_model

            if not api_key or "your-" in api_key:
                logger.warning(
                    "[LLMClient] OpenAI API key not set or placeholder detected. "
                    "Running in simulated Mock Mode for demonstration."
                )
                self.is_mock = True
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"[LLMClient] Connected to OpenAI with model '{self.chat_model}'")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'gemini', 'openai' or 'azure'.")

    @staticmethod
    def is_placeholder_key(api_key: Optional[str]) -> bool:
        """True if the API key is missing or still the placeholder from .env.example."""
        return not api_key or "your-" in api_key

    def classify_email(self, system_prompt: str, user_prompt: str) -> LLMClassificationResponse:
        """
        Classifies the email into multi-label categories using structured output.
        """
        if self.is_mock:
            return self._mock_classify_email(user_prompt)

        try:
            # Using OpenAI / Azure OpenAI structured output parsing
            response = self.client.beta.chat.completions.parse(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=LLMClassificationResponse,
                **self._sampling_params,
            )
            return response.choices[0].message.parsed

        except Exception as e:
            # Fallback to standard json_object format if parse is not supported on older deployment
            logger.warning(f"Structured parse failed ({e}), attempting JSON object fallback...")
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                system_prompt
                                + "\nYou must respond strictly in JSON matching this JSON schema:\n"
                                + json.dumps(LLMClassificationResponse.model_json_schema())
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    **self._sampling_params,
                )
                raw_json = response.choices[0].message.content
                data = json.loads(raw_json)
                return LLMClassificationResponse(**data)
            except Exception as inner_e:
                logger.error(f"Error calling LLM: {inner_e}")
                raise inner_e

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """
        Uses the vision-capable chat model to transcribe text and describe an image.
        """
        if self.is_mock:
            return self._mock_describe_image(image_bytes)

        try:
            b64_image = base64.b64encode(image_bytes).decode("ascii")
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You read images attached to customer emails. Transcribe all visible text exactly, "
                            "then briefly describe anything relevant such as screenshots, error dialogs, tables "
                            "or charts. Respond with plain text only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                        ],
                    },
                ],
                **self._sampling_params,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            raise e

    def _mock_classify_email(self, user_prompt: str) -> LLMClassificationResponse:
        """
        Deterministic mock generator for demonstration when API keys are not provided.
        Simulates multi-label predictions and confidence scores based on text contents.
        """
        lower = user_prompt.lower()
        predicted: List[PredictedClass] = []

        # Check for billing keywords
        if any(w in lower for w in ["invoice", "charge", "refund", "billing", "payment", "receipt", "overcharge"]):
            if "refund" in lower or "cancel" in lower:
                predicted.append(PredictedClass(
                    class_name="Cancellation & Refund Request",
                    confidence=0.88,
                    reasoning="Customer explicitly mentions refund or cancellation regarding their invoice charges."
                ))
            predicted.append(PredictedClass(
                class_name="Billing & Invoicing",
                confidence=0.92,
                reasoning="Email and attachment discuss invoice discrepancy and payment issues."
            ))

        # Check for complaint / escalation
        if any(w in lower for w in ["unacceptable", "furious", "terrible", "escalate", "unresolved", "manager", "frustrated"]):
            predicted.append(PredictedClass(
                class_name="Complaint & Escalation",
                confidence=0.82,
                reasoning="Strong expressions of frustration, dissatisfaction, and escalation demand."
            ))

        # Check for tech support
        if any(w in lower for w in ["error", "bug", "crash", "failed", "outage", "timeout", "not working", "500", "404"]):
            predicted.append(PredictedClass(
                class_name="Technical Support / Bug Report",
                confidence=0.86,
                reasoning="Mentions system error code, malfunction, or unexpected behavior."
            ))

        # Check for account / security
        if any(w in lower for w in ["password", "login", "2fa", "security", "unauthorized", "locked", "sso"]):
            predicted.append(PredictedClass(
                class_name="Account & Security Management",
                confidence=0.89,
                reasoning="Concerns user credentials, login trouble, or security settings."
            ))

        # Check for sales / product
        if any(w in lower for w in ["quote", "demo", "pricing", "enterprise", "features", "sales"]):
            predicted.append(PredictedClass(
                class_name="Product Inquiry & Sales",
                confidence=0.85,
                reasoning="Potential customer asking for product demo and enterprise pricing."
            ))

        # Default fallback if nothing triggered
        if not predicted:
            predicted.append(PredictedClass(
                class_name="General Inquiry / Feedback",
                confidence=0.68,
                reasoning="General communication without clear single department indicators."
            ))

        return LLMClassificationResponse(
            primary_intent="User request parsed by simulated LLM layer.",
            predicted_classes=predicted,
            summary="Processed email and associated attachments for multi-label assignment."
        )

    def _mock_describe_image(self, image_bytes: bytes) -> str:
        """
        Mock image description for demonstration when API keys are not provided.
        """
        return f"[Mock image description: image of {len(image_bytes)} bytes]"
