"""
Text Segment Module

This module provides structures for representing text segments
with hierarchical relationships.
"""

from typing import Dict, List, Optional, Any, Union
import uuid
from dataclasses import dataclass, field


@dataclass
class Segment:
    """
    Represents a segment of text with hierarchical structure.
    
    A segment can contain subsegments, creating a tree-like structure
    that can represent various levels of text organization (e.g., sections,
    paragraphs, sentences).
    
    Attributes:
        id: Unique identifier for the segment
        text: Text content of the segment
        segment_type: Type of segment (e.g. "section", "paragraph", "sentence")
        level: Hierarchical level (0 for top-level, increasing for deeper levels)
        parent_id: ID of the parent segment, if any
        position: Position in the original text (start, end)
        metadata: Additional information about the segment
        children: List of child segments
    """
    
    text: str
    segment_type: str
    level: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    position: Dict[str, int] = field(default_factory=lambda: {"start": 0, "end": 0})
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['Segment'] = field(default_factory=list)
    
    def add_child(self, segment: 'Segment') -> None:
        """
        Add a child segment to this segment.
        
        Args:
            segment: Child segment to add
        """
        segment.parent_id = self.id
        segment.level = self.level + 1
        self.children.append(segment)
    
    def find_segment(self, segment_id: str) -> Optional['Segment']:
        """
        Find a segment by ID in this segment's hierarchy.
        
        Args:
            segment_id: ID of the segment to find
            
        Returns:
            Optional[Segment]: Found segment or None
        """
        if self.id == segment_id:
            return self
            
        for child in self.children:
            found = child.find_segment(segment_id)
            if found:
                return found
                
        return None
    
    def get_path(self) -> List[str]:
        """
        Get the path of segment IDs from the root to this segment.
        
        Returns:
            List[str]: List of segment IDs
        """
        if not self.parent_id:
            return [self.id]
            
        # Need to find the parent by traversing up from root
        # This is a simplification - in practice would need to use a global registry
        # or pass the root segment
        return []
    
    def get_hierarchical_id(self, separator: str = ".") -> str:
        """
        Get a hierarchical ID based on the segment's position in the tree.
        
        For example: "1.2.3" for the 3rd segment of the 2nd subsection of the 1st section.
        
        Args:
            separator: Character to use as separator
            
        Returns:
            str: Hierarchical ID
        """
        # This is a simplified placeholder - would need to know position among siblings
        # In practice, would need to calculate based on the full tree structure
        return f"{self.level}{separator}{self.id[:4]}"
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the contextual information for this segment.
        
        This includes information about the segment's place in the hierarchy,
        its text, and any metadata.
        
        Returns:
            Dict[str, Any]: Context dictionary
        """
        return {
            "id": self.id,
            "hierarchical_id": self.get_hierarchical_id(),
            "text": self.text,
            "segment_type": self.segment_type,
            "level": self.level,
            "parent_id": self.parent_id,
            "position": self.position,
            "metadata": self.metadata,
            "has_children": len(self.children) > 0,
            "num_children": len(self.children)
        }
    
    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """
        Convert the segment to a dictionary representation.
        
        Args:
            include_children: Whether to include children in the dictionary
            
        Returns:
            Dict[str, Any]: Dictionary representation of the segment
        """
        result = {
            "id": self.id,
            "text": self.text,
            "segment_type": self.segment_type,
            "level": self.level,
            "parent_id": self.parent_id,
            "position": self.position,
            "metadata": self.metadata
        }
        
        if include_children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Segment':
        """
        Create a segment from a dictionary representation.
        
        Args:
            data: Dictionary representation of a segment
            
        Returns:
            Segment: Created segment
        """
        children_data = data.pop("children", [])
        segment = Segment(**data)
        
        for child_data in children_data:
            child = Segment.from_dict(child_data)
            segment.add_child(child)
            
        return segment


@dataclass
class SegmentationResult:
    """
    Represents the results of text segmentation.
    
    Attributes:
        root: Root segment containing the full hierarchy
        segments: Flat list of all segments
        metadata: Additional information about the segmentation
    """
    
    root: Segment
    segments: List[Segment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize segments list if not provided."""
        if not self.segments:
            self.collect_segments(self.root)
    
    def collect_segments(self, segment: Segment) -> None:
        """
        Collect all segments from the hierarchy into a flat list.
        
        Args:
            segment: Starting segment
        """
        self.segments.append(segment)
        for child in segment.children:
            self.collect_segments(child)
    
    def find_segment(self, segment_id: str) -> Optional[Segment]:
        """
        Find a segment by ID.
        
        Args:
            segment_id: ID of the segment to find
            
        Returns:
            Optional[Segment]: Found segment or None
        """
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the segmentation result to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "root": self.root.to_dict(),
            "metadata": self.metadata,
            "segment_count": len(self.segments)
        } 