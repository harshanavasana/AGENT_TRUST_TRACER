from agent.voice_commands import parse_voice_command


def test_voice_command_normalizes_and_recognizes_youtube() -> None:
    command = parse_voice_command("  Open   YouTube for me  ")

    assert command.text == "Open YouTube for me"
    assert command.destination == "youtube"
    assert command.url == "https://www.youtube.com/"


def test_voice_command_recognizes_gmail_case_insensitively() -> None:
    command = parse_voice_command("please take me to GMAIL")

    assert command.destination == "gmail"


def test_voice_command_does_not_turn_arbitrary_urls_into_navigation() -> None:
    command = parse_voice_command("Find the latest project update")

    assert command.destination is None
    assert command.url is None


def test_destination_word_without_browser_action_is_not_executed() -> None:
    command = parse_voice_command("Search my email about YouTube")

    assert command.destination is None
    assert command.url is None


def test_natural_language_navigation_is_recognized() -> None:
    command = parse_voice_command("Could you take me to the Gmail website please?")

    assert command.destination == "gmail"


def test_voice_command_infers_youtube_source_for_research() -> None:
    command = parse_voice_command("Find the latest video about robotics")

    assert command.source == "YouTube"


def test_voice_command_infers_gmail_source_for_research() -> None:
    command = parse_voice_command("Find my unread project emails")

    assert command.source == "Gmail"
