"""
XML Output Parser - Extract content from XML-style tags in LLM output.

This module provides functionality to parse and extract content from XML-style tags
(e.g., <construction>, <think>) in LLM output text. It uses a regex-first approach
with XML parser fallback for robustness.
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
from html import unescape

logger = logging.getLogger(__name__)


class XMLOutputParser:
    """
    Parser for extracting content from XML-style tags in LLM output.

    This parser is designed to handle the 2-phase meta-prompt system where LLMs
    output structured responses with construction phase in <construction> tags
    and reasoning phase in <think> tags.

    Features:
    - Regex-based extraction (handles malformed XML)
    - XML parser fallback for well-formed XML
    - Support for nested tags, attributes, and multiple occurrences
    - Comprehensive error handling and logging
    """

    @staticmethod
    def extract(text: str, tag_name: str, preserve_whitespace: bool = False) -> Dict[str, Any]:
        """
        Extract content from XML-style tag in text.

        Args:
            text: Raw text containing XML tags
            tag_name: Name of the tag to extract (e.g., "construction", "think")
            preserve_whitespace: If True, preserve whitespace/newlines; if False, strip them

        Returns:
            Dictionary containing:
                - success: bool - Whether extraction succeeded
                - content: str - Extracted content (empty string on failure)
                - metadata: dict - Additional information about extraction
                - error: str (optional) - Error message if success=False

        Raises:
            ValueError: If inputs are invalid (empty text, invalid tag name)

        Example:
            >>> parser = XMLOutputParser()
            >>> result = parser.extract(
            ...     "Text before <construction>entities: user</construction> after",
            ...     "construction"
            ... )
            >>> result["success"]
            True
            >>> result["content"]
            'entities: user'
        """
        # Input validation
        if not text:
            error_msg = "Input text cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not tag_name:
            error_msg = "Tag name cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate tag name (alphanumeric, underscore, hyphen only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', tag_name):
            error_msg = f"Invalid tag name: '{tag_name}'. Must be alphanumeric with underscore/hyphen only"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Extracting tag '{tag_name}' from text (length: {len(text)})")

        # Try regex extraction first (handles malformed XML)
        regex_result = XMLOutputParser._extract_with_regex(text, tag_name, preserve_whitespace)

        if regex_result["success"]:
            logger.info(
                f"Successfully extracted tag '{tag_name}' using regex",
                extra={
                    "tag_name": tag_name,
                    "content_length": len(regex_result["content"]),
                    "num_occurrences": regex_result["metadata"]["num_occurrences"]
                }
            )
            return regex_result

        # Fallback to XML parser for well-formed XML
        logger.debug(f"Regex extraction failed, trying XML parser fallback")
        xml_result = XMLOutputParser._extract_with_xml_parser(text, tag_name, preserve_whitespace)

        if xml_result["success"]:
            logger.info(
                f"Successfully extracted tag '{tag_name}' using XML parser",
                extra={
                    "tag_name": tag_name,
                    "content_length": len(xml_result["content"])
                }
            )
            return xml_result

        # Both methods failed
        error_msg = f"Tag '{tag_name}' not found in text"
        logger.warning(
            error_msg,
            extra={
                "tag_name": tag_name,
                "text_length": len(text),
                "text_preview": text[:100] if len(text) > 100 else text
            }
        )

        return {
            "success": False,
            "content": "",
            "error": error_msg,
            "metadata": {
                "tag_found": False,
                "num_occurrences": 0,
                "extraction_method": None
            }
        }

    @staticmethod
    def _extract_with_regex(
        text: str,
        tag_name: str,
        preserve_whitespace: bool
    ) -> Dict[str, Any]:
        """
        Extract content using regex pattern.

        Handles:
        - Multiline content (re.DOTALL)
        - Tags with attributes
        - Nested tags (extracts outermost)
        - Multiple occurrences (takes first, logs warning)

        Args:
            text: Input text
            tag_name: Tag to extract
            preserve_whitespace: Whether to preserve whitespace

        Returns:
            Extraction result dictionary
        """
        # Pattern handles tags with optional attributes: <tag attr="value">content</tag>
        pattern = rf'<{tag_name}(?:\s+[^>]*)?>(.*?)</{tag_name}>'

        try:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

            if not matches:
                return {
                    "success": False,
                    "content": "",
                    "metadata": {
                        "tag_found": False,
                        "num_occurrences": 0,
                        "extraction_method": "regex"
                    }
                }

            # Handle multiple occurrences
            num_occurrences = len(matches)
            if num_occurrences > 1:
                logger.warning(
                    f"Found {num_occurrences} occurrences of tag '{tag_name}', using first occurrence",
                    extra={"tag_name": tag_name, "num_occurrences": num_occurrences}
                )

            # Extract first occurrence
            content = matches[0]

            # Unescape XML entities (e.g., &lt; -> <, &amp; -> &)
            content = unescape(content)

            # Strip or preserve whitespace
            if not preserve_whitespace:
                content = content.strip()

            return {
                "success": True,
                "content": content,
                "metadata": {
                    "tag_found": True,
                    "num_occurrences": num_occurrences,
                    "extraction_method": "regex"
                }
            }

        except re.error as e:
            logger.error(
                f"Regex error during extraction: {e}",
                exc_info=True,
                extra={"tag_name": tag_name}
            )
            return {
                "success": False,
                "content": "",
                "error": f"Regex error: {str(e)}",
                "metadata": {
                    "tag_found": False,
                    "num_occurrences": 0,
                    "extraction_method": "regex"
                }
            }

    @staticmethod
    def _extract_with_xml_parser(
        text: str,
        tag_name: str,
        preserve_whitespace: bool
    ) -> Dict[str, Any]:
        """
        Extract content using XML parser (fallback method).

        More robust for well-formed XML with complex nesting.

        Args:
            text: Input text
            tag_name: Tag to extract
            preserve_whitespace: Whether to preserve whitespace

        Returns:
            Extraction result dictionary
        """
        try:
            # Wrap text in root element to make it valid XML
            wrapped_text = f"<root>{text}</root>"
            root = ET.fromstring(wrapped_text)

            # Find the target tag
            element = root.find(f".//{tag_name}")

            if element is None:
                return {
                    "success": False,
                    "content": "",
                    "metadata": {
                        "tag_found": False,
                        "num_occurrences": 0,
                        "extraction_method": "xml_parser"
                    }
                }

            # Get text content (includes nested tags' text)
            content = ET.tostring(element, encoding='unicode', method='text')

            # Unescape XML entities
            content = unescape(content)

            # Strip or preserve whitespace
            if not preserve_whitespace:
                content = content.strip()

            # Count occurrences
            all_elements = root.findall(f".//{tag_name}")
            num_occurrences = len(all_elements)

            if num_occurrences > 1:
                logger.warning(
                    f"Found {num_occurrences} occurrences of tag '{tag_name}', using first occurrence",
                    extra={"tag_name": tag_name, "num_occurrences": num_occurrences}
                )

            return {
                "success": True,
                "content": content,
                "metadata": {
                    "tag_found": True,
                    "num_occurrences": num_occurrences,
                    "extraction_method": "xml_parser"
                }
            }

        except ET.ParseError as e:
            logger.debug(
                f"XML parsing failed: {e}",
                extra={"tag_name": tag_name, "error": str(e)}
            )
            return {
                "success": False,
                "content": "",
                "metadata": {
                    "tag_found": False,
                    "num_occurrences": 0,
                    "extraction_method": "xml_parser"
                }
            }
        except Exception as e:
            logger.error(
                f"Unexpected error in XML parser: {e}",
                exc_info=True,
                extra={"tag_name": tag_name}
            )
            return {
                "success": False,
                "content": "",
                "error": f"XML parser error: {str(e)}",
                "metadata": {
                    "tag_found": False,
                    "num_occurrences": 0,
                    "extraction_method": "xml_parser"
                }
            }


# Convenience functions
def extract_construction(text: str, preserve_whitespace: bool = False) -> Dict[str, Any]:
    """
    Extract content from <construction> tag.

    Args:
        text: Raw LLM output text
        preserve_whitespace: Whether to preserve whitespace

    Returns:
        Extraction result dictionary
    """
    return XMLOutputParser.extract(text, "construction", preserve_whitespace)


def extract_reasoning(text: str, preserve_whitespace: bool = False) -> Dict[str, Any]:
    """
    Extract content from <think> tag.

    Args:
        text: Raw LLM output text
        preserve_whitespace: Whether to preserve whitespace

    Returns:
        Extraction result dictionary
    """
    return XMLOutputParser.extract(text, "think", preserve_whitespace)
