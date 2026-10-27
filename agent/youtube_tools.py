"""Read-only public YouTube Data API tools for the research agent."""

from __future__ import annotations

import os
from typing import Any

MAX_YOUTUBE_RESULTS = 5
MAX_YOUTUBE_QUERY_CHARS = 200
MAX_YOUTUBE_DESCRIPTION_CHARS = 240


def build_youtube_service(api_key: str | None = None) -> Any:
    """Create a YouTube Data API client from the local API key."""
    from googleapiclient.discovery import build

    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("Set YOUTUBE_API_KEY in your local .env file.")
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def build_youtube_tools(service: Any) -> list[Any]:
    """Build public search/details tools; channel lookup is limited to search results."""
    from langchain_core.tools import tool

    searched_channel_ids: set[str] = set()

    @tool
    def search_youtube(query: str) -> str:
        """Search public YouTube videos and return up to five video summaries."""
        query = query.strip()
        if not query:
            raise ValueError("Enter a non-empty YouTube search query.")
        if len(query) > MAX_YOUTUBE_QUERY_CHARS:
            raise ValueError(
                f"YouTube search queries must be at most "
                f"{MAX_YOUTUBE_QUERY_CHARS} characters."
            )

        response = (
            service.search()
            .list(
                part="snippet",
                q=query,
                type="video",
                maxResults=MAX_YOUTUBE_RESULTS,
            )
            .execute()
        )
        items = response.get("items", [])
        if not items:
            return "No public YouTube videos matched that search."

        video_ids = []
        video_channels: dict[str, str] = {}
        for item in items:
            identifier = item.get("id", {})
            video_id = identifier.get("videoId")
            snippet = item.get("snippet", {})
            channel_id = snippet.get("channelId")
            if video_id:
                video_ids.append(video_id)
                if channel_id:
                    searched_channel_ids.add(channel_id)
                    video_channels[video_id] = channel_id

        details_by_id: dict[str, dict[str, Any]] = {}
        if video_ids:
            details = (
                service.videos()
                .list(
                    part="contentDetails,statistics",
                    id=",".join(video_ids),
                    maxResults=MAX_YOUTUBE_RESULTS,
                )
                .execute()
            )
            details_by_id = {
                item["id"]: item for item in details.get("items", [])
            }

        summaries = []
        for item in items:
            identifier = item.get("id", {})
            video_id = identifier.get("videoId")
            if not video_id:
                continue
            snippet = item.get("snippet", {})
            detail = details_by_id.get(video_id, {})
            statistics = detail.get("statistics", {})
            summaries.append(
                f"Video ID: {video_id}\n"
                f"Title: {snippet.get('title', '(untitled)')}\n"
                f"Channel: {snippet.get('channelTitle', '(unknown)')}\n"
                f"Channel ID: {video_channels.get(video_id, '(unknown)')}\n"
                f"Published: {snippet.get('publishedAt', '(unknown)')}\n"
                f"Duration: {detail.get('contentDetails', {}).get('duration', '(unknown)')}\n"
                f"Views: {statistics.get('viewCount', '(unknown)')}\n"
                f"Description: {snippet.get('description', '')[:MAX_YOUTUBE_DESCRIPTION_CHARS]}"
            )
        return "\n\n".join(summaries) or "No videos with public details were returned."

    @tool
    def get_youtube_channel_details(channel_id: str) -> str:
        """Get public statistics for a channel ID returned by search_youtube."""
        if channel_id not in searched_channel_ids:
            raise ValueError(
                "Search YouTube first and use a channel ID from those results."
            )
        response = (
            service.channels()
            .list(
                part="snippet,statistics",
                id=channel_id,
                maxResults=1,
            )
            .execute()
        )
        channels = response.get("items", [])
        if not channels:
            return "No public channel details were found."
        channel = channels[0]
        snippet = channel.get("snippet", {})
        statistics = channel.get("statistics", {})
        return (
            f"Channel: {snippet.get('title', '(unknown)')}\n"
            f"Published: {snippet.get('publishedAt', '(unknown)')}\n"
            f"Subscribers: {statistics.get('subscriberCount', '(hidden or unavailable)')}\n"
            f"Videos: {statistics.get('videoCount', '(unknown)')}\n"
            f"Views: {statistics.get('viewCount', '(unknown)')}\n"
            f"Description: "
            f"{snippet.get('description', '')[:MAX_YOUTUBE_DESCRIPTION_CHARS]}"
        )

    return [search_youtube, get_youtube_channel_details]
