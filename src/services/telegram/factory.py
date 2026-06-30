import logging
from typing import Optional

from src.config import get_settings
from src.services.telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def make_telegram_service(
    opensearch_client,
    embeddings_client,
    ollama_client,
    cache_client=None,
    langfuse_tracer=None,
) -> Optional[TelegramBot]:
    """
    Create Telegram bot if enabled.
    """
    # PYTHON: pulls the cached Settings singleton (Pydantic Settings object,
    # built once from env vars).
    # JAVA COMPARISON: like @Autowired private TelegramSettings settings; —
    # except here it's an explicit function call instead of field injection,
    # because this project doesn't use a DI container (no Spring context).
    # Every "factory" function in this codebase IS the DI mechanism.
    settings = get_settings()

    # Feature flag check — TELEGRAM__ENABLED in .env
    # JAVA: like @ConditionalOnProperty(name = "telegram.enabled", havingValue = "true")
    # on a @Bean method — Spring would just skip creating the bean.
    # Here, the function explicitly returns None instead, and the caller
    # (main.py) has to check for that None.
    if not settings.telegram.enabled:
        logger.info("Telegram bot is disabled")
        return None

    # Guard against "enabled=true but forgot to set the token" misconfig.
    # JAVA: like a @PostConstruct validation check that logs a warning
    # and leaves the bean uninitialized, rather than throwing and crashing
    # the whole application context.
    if not settings.telegram.bot_token:
        logger.warning("Telegram bot token not configured")
        return None

    # Construction — straightforward dependency injection by constructor args.
    # JAVA: equivalent to
    #   new TelegramBot(settings.getTelegram().getBotToken(),
    #                    opensearchClient, embeddingsClient, ollamaClient, cacheClient)
    # — this whole function IS effectively a @Bean factory method in a
    # Spring @Configuration class, just without the annotation magic.
    bot = TelegramBot(
        bot_token=settings.telegram.bot_token,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        ollama_client=ollama_client,
        cache_client=cache_client,
    )

    logger.info("Telegram bot created successfully")
    return bot
