from pydantic import BaseModel, Field

class BookMetadata(BaseModel):
    first_publish_year: int | None = Field(default=None, description="The year the book was first published")
    total_editions_cataloged: int = Field(default=0, description="Total number of editions cataloged for the book")
    top_subjects: list[str] = Field(default=[], description="Top subjects or genres associated with the book")
    link: str = Field(default="", description="Link to the book's page on Open Library")

class BookCandidate(BaseModel):
    title: str = Field(default="", description="The title of the book")
    authors: list = Field(default=[], description="Authors of the book")
    meta: BookMetadata = Field(default=None, description="Metadata about the book")