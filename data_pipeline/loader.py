import json
import os
import re
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from config.settings import settings
from config.app_config import config

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    get_leaf_nodes
)
from llama_index.core.schema import Document, TextNode, BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI


class AdvancedDocumentLoader:
    def __init__(
            self,
            input_dir: str = "data",
            chunk_sizes: List[int] = [2048, 1024, 512],
            embed_model: str = settings.EMBEDDING_MODEL,
            parsing_mode: str = "hierarchical" # hierarchical or semantic
    ):
        self.input_dir = input_dir
        self.chunk_sizes = chunk_sizes
        self.parsing_mode =  parsing_mode
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model

        self.embed_model = OpenAIEmbedding(model=embed_model)

        # Initialize OpenAI client for summarization
        self.openai_client = AsyncOpenAI(
            api_key="tQPcEbKybXQAUkI60QoeF5pmC63EuvmOMoA8UjZfKREbgDfVnInzu1FxkAIxP1T3BlbkFJ6rhONZM-0RTT3Z8eoW9rJEDDO7vtjNH36sSr5h8S4pmT_E-l4FIV5Kr1-Q40aujHQuRXaYmA8A",
            timeout=30,  # Increased timeout
            max_retries=3
            )

        self.conversation_metadata_fields = [
            "participants",
            "discussion_topics",
            "investment_stage",
            "key_metrics"
        ]

    async def _extract_conversation_metadata(self, text: str, file_name: str) -> Dict[str, Any]:
        """Extract conversation-specific metadata"""
        system_prompt = """You are an AI that analyzes investment conversations between companies and investors.
        Extract the following metadata:
        - participants: List of participants and their roles (investor/company)
        - discussion_topics: Key topics discussed (funding, valuation, terms, etc.)
        - investment_stage: Current stage (seed, Series A, etc.) if mentioned
        - key_metrics: Any important metrics mentioned (valuation, revenue, etc.)
        
        Return JSON with these fields. Be concise and accurate."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{text[:8000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            metadata = json.loads(response.choices[0].message.content)

            # Validate we got all required fields
            for field in self.conversation_metadata_fields:
                if field not in metadata:
                    metadata[field] = None

            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting conversation metadata: {e}")
            return {field: None for field in self.conversation_metadata_fields}

    def _load_raw_documents(self) -> List[Document]:
        """Use default file readers to load documents from directory"""
        try:
            raw_docs = SimpleDirectoryReader(self.input_dir).load_data()
            self.logger.info(f"Loaded {len(raw_docs)} documents from {self.input_dir}")
            return raw_docs
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return []

    def _parse_nodes(self, documents: List[Document]) -> List[BaseNode]:
        """Parse documents into nodes using selected strategy"""
        if self.parsing_mode == "semantic":
            # Semantic parsing keeps related content together
            parser = SemanticSplitterNodeParser(
                buffer_size=1,  # How much to group by similarity
                embed_model=self.embed_model,
                breakpoint_percentile_threshold=95
            )
            base_nodes = parser.get_nodes_from_documents(documents)
            return get_leaf_nodes(base_nodes)
        else:
            # Default to hierarchical parsing
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=self.chunk_sizes
            )
            return parser.get_nodes_from_documents(documents)

    def _create_document(
            self,
            text: str,
            file_name: str,
            **extra_metadata
    ) -> Document:
        """Create a Document with standardized metadata"""
        metadata = {
            "source": "document",
            "file_name": file_name,
            "file_type": os.path.splitext(file_name)[1][1:],
            "processing_time": datetime.now(timezone.utc).isoformat(),
            **extra_metadata
        }

        return Document(text=text, metadata=metadata)

    async def _generate_summary(self, text: str) -> Dict[str, str]:
        """Generate title and summary for a document chunk"""
        system_prompt = """You are an AI that extracts titles and summarizes content.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Create a concise, descriptive title for this content.
        For the summary: Create a 1-3 sentence summary of the main points.
        Be factual and maintain the original meaning."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{text[:8000]}"}  # Increased context window
                ],
                response_format={"type": "json_object"},
                temperature=0.2  # More deterministic output
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"title": "Untitled", "summary": "No summary available"}

    async def _enhance_nodes(self, nodes: List[BaseNode]) -> List[TextNode]:
        """Add embeddings and enhanced metadata to nodes"""
        enhanced_nodes = []

        for node in nodes:
            try:
                # Generate summary for the node
                summary_data = await self._generate_summary(node.get_content())

                # Create enhanced metadata
                enhanced_metadata = {
                    **node.metadata,
                    "title": summary_data["title"],
                    "summary": summary_data["summary"],
                    "content_length": len(node.get_content()),
                    "node_type": type(node).__name__,
                    "enhanced_at": datetime.now(timezone.utc).isoformat(),
                    "processing_strategy": self.parsing_mode
                }

                # Get embedding for the node
                embedding = await self.embed_model.aget_text_embedding(node.get_content())

                # Create enhanced TextNode
                enhanced_node = TextNode(
                    text=node.get_content(),
                    embedding=embedding,
                    metadata=enhanced_metadata,
                    excluded_embed_metadata_keys=["file_name", "file_type"],  # Don't embed these
                    excluded_llm_metadata_keys=["processing_time"]  # Don't send these to LLM
                )

                enhanced_nodes.append(enhanced_node)

            except Exception as e:
                self.logger.error(f"Error enhancing node: {e}")
                continue

        return enhanced_nodes

    async def load_and_process(self) -> List[TextNode]:
        """Load, parse, and enhance documents"""
        try:
            # Load raw documents
            raw_docs = self._load_raw_documents()
            all_docs = []

            # Create standardized documents with metadata
            for doc in raw_docs:
                file_name = doc.metadata.get("file_name", "unknown")
                # Extract conversation-specific metadata
                conversation_metadata = await self._extract_conversation_metadata(doc.text, file_name)

                # Create document with all metadata
                all_docs.append(self._create_document(
                    text=doc.text,
                    file_name=file_name,
                    **conversation_metadata
                ))

            # Parse into nodes
            nodes = self._parse_nodes(all_docs)
            # Enhance nodes with embeddings and metadata
            enhanced_nodes = await self._enhance_nodes(nodes)

            return enhanced_nodes

        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            raise