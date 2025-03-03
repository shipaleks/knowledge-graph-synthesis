"""
Text Segment Module

This module provides classes for representing and managing
text segments as part of the text processing pipeline.
"""

import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Segment:
    """
    Represents a segment of text with metadata.
    
    A segment can be a section, subsection, paragraph, list, table, etc.
    """
    id: str
    text: str
    segment_type: str  # section, subsection, paragraph, list, table, etc.
    level: int = 1
    title: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_child(self, child_id: str) -> None:
        """
        Add a child segment ID to this segment.
        
        Args:
            child_id: ID of the child segment
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def set_parent(self, parent_id: str) -> None:
        """
        Set the parent segment ID for this segment.
        
        Args:
            parent_id: ID of the parent segment
        """
        self.parent_id = parent_id
    
    def get_hierarchical_id(self) -> str:
        """
        Get a hierarchical ID representing the segment's position.
        
        Returns:
            str: Hierarchical ID (e.g., "1.2.3")
        """
        if self.metadata.get("hierarchical_id"):
            return self.metadata["hierarchical_id"]
        return self.id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the segment to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the segment
        """
        return {
            "id": self.id,
            "text": self.text,
            "segment_type": self.segment_type,
            "level": self.level,
            "title": self.title,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Segment':
        """
        Create a segment from a dictionary.
        
        Args:
            data: Dictionary representation of the segment
            
        Returns:
            Segment: New segment instance
        """
        return cls(
            id=data["id"],
            text=data["text"],
            segment_type=data["segment_type"],
            level=data.get("level", 1),
            title=data.get("title"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {})
        )


class SegmentationResult:
    """
    Contains the results of text segmentation.
    
    Provides access to segments and their relationships.
    """
    
    def __init__(self, segments: Optional[List[Segment]] = None):
        """
        Initialize the segmentation result.
        
        Args:
            segments: List of segments
        """
        self.segments = segments or []
        self._segments_by_id = {segment.id: segment for segment in self.segments}
    
    def add_segment(self, segment: Segment) -> None:
        """
        Add a segment to the result.
        
        Args:
            segment: Segment to add
        """
        self.segments.append(segment)
        self._segments_by_id[segment.id] = segment
    
    def get_segment(self, segment_id: str) -> Optional[Segment]:
        """
        Get a segment by ID.
        
        Args:
            segment_id: ID of the segment
            
        Returns:
            Optional[Segment]: Segment or None if not found
        """
        return self._segments_by_id.get(segment_id)
    
    def get_children(self, segment_id: str) -> List[Segment]:
        """
        Get all child segments for a segment.
        
        Args:
            segment_id: ID of the parent segment
            
        Returns:
            List[Segment]: List of child segments
        """
        parent = self.get_segment(segment_id)
        if not parent:
            return []
        
        return [self.get_segment(child_id) for child_id in parent.children_ids
                if self.get_segment(child_id)]
    
    def get_parent(self, segment_id: str) -> Optional[Segment]:
        """
        Get the parent segment for a segment.
        
        Args:
            segment_id: ID of the child segment
            
        Returns:
            Optional[Segment]: Parent segment or None
        """
        segment = self.get_segment(segment_id)
        if not segment or not segment.parent_id:
            return None
        
        return self.get_segment(segment.parent_id)
    
    def build_hierarchy(self) -> Dict[str, Any]:
        """
        Build a hierarchical structure of segments.
        
        Returns:
            Dict[str, Any]: Hierarchical structure
        """
        root_segments = [s for s in self.segments if not s.parent_id]
        
        def build_subtree(segment: Segment) -> Dict[str, Any]:
            children = self.get_children(segment.id)
            result = segment.to_dict()
            
            if children:
                result["children"] = [build_subtree(child) for child in children]
            
            return result
        
        return {
            "segments": [build_subtree(root) for root in root_segments]
        }
    
    def to_json(self, pretty: bool = True) -> str:
        """
        Convert the segmentation result to JSON.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation
        """
        hierarchy = self.build_hierarchy()
        indent = 2 if pretty else None
        return json.dumps(hierarchy, ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentationResult':
        """
        Create a segmentation result from a dictionary.
        
        Args:
            data: Dictionary representation of the segmentation result
            
        Returns:
            SegmentationResult: New segmentation result instance
        """
        result = cls()
        
        # Helper function to recursively process segments
        def process_segment(segment_data: Dict[str, Any]) -> None:
            # Create segment from data
            segment = Segment.from_dict(segment_data)
            result.add_segment(segment)
            
            # Process children if any
            for child_data in segment_data.get("children", []):
                child = Segment.from_dict(child_data)
                child.parent_id = segment.id
                segment.add_child(child.id)
                result.add_segment(child)
                
                # Process nested children recursively
                if "children" in child_data:
                    for grandchild_data in child_data["children"]:
                        process_segment_with_parent(grandchild_data, child.id)
        
        # Helper function to process a segment with a known parent
        def process_segment_with_parent(segment_data: Dict[str, Any], parent_id: str) -> None:
            segment = Segment.from_dict(segment_data)
            segment.parent_id = parent_id
            result.add_segment(segment)
            
            # Add as child to parent
            parent = result.get_segment(parent_id)
            if parent:
                parent.add_child(segment.id)
            
            # Process children recursively
            for child_data in segment_data.get("children", []):
                process_segment_with_parent(child_data, segment.id)
        
        # Process all root segments
        if "segments" in data:
            for segment_data in data["segments"]:
                process_segment(segment_data)
        
        return result
    
    def __len__(self) -> int:
        """
        Get the number of segments.
        
        Returns:
            int: Number of segments
        """
        return len(self.segments)


def create_segment(
    text: str, 
    segment_type: str, 
    level: int = 1, 
    title: Optional[str] = None, 
    parent_id: Optional[str] = None
) -> Segment:
    """
    Create a new segment.
    
    Args:
        text: Segment text content
        segment_type: Type of segment (section, paragraph, etc.)
        level: Hierarchical level
        title: Optional segment title
        parent_id: Optional parent segment ID
        
    Returns:
        Segment: New segment instance
    """
    segment_id = str(uuid.uuid4())
    
    # Create segment
    segment = Segment(
        id=segment_id,
        text=text,
        segment_type=segment_type,
        level=level,
        title=title,
        parent_id=parent_id
    )
    
    return segment 