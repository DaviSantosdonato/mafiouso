#requirements:
#requests
#beautifulsoup4
#google-genai
#wmill
"""Ofertas pedevapor.shop -> copy de vendas com Gemini -> WhatsApp via Evolution API.

Alvo: pedevapor.shop (Nuxt + tema `/theme/varejo/`). O HTML dos cards JÁ vem
renderizado no servidor (`div.pcard`), por isso o modo padrão (`ENGINE = "auto"`)
tenta o Playwright e cai para `requests` quando o navegador não está disponível.
Para forçar headless Chrome, passe `engine="playwright"`.

Ordem do pipeline:
    1. coleta os cards de produto (somente com estoque)
    2. converte a URL do produto em link de afiliado (?ref=VSSSSLPBCD)
    3. pede ao Gemini o título limpo + gatilho (JSON) -> fallback heurístico
    4. monta a mensagem de WhatsApp no template fixo do negócio
    5. devolve itens no formato n8n: [{"json": {...}}, ...]

Chave do Gemini: parâmetro -> env GEMINI_API_KEY -> variável f/ofertas/gemini_api_key.

Rodar local: `python script.py` (imprime o JSON).
"""

import json
import os
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Comment, NavigableString

# ==========================================================================
# PARÂMETROS DO NEGÓCIO (isole tudo aqui)
# ==========================================================================
BASE_AFFILIATE_URL = "https://pedevapor.shop"
TARGET_URL = "https://pedevapor.shop"
AFFILIATE_REF = "VSSSSLPBCD"

# O modelo pedido originalmente ("gemini-2.5-flash-lite") passou a responder
# 404 NOT_FOUND para chaves novas ("no longer available to new users").
# "gemini-3.5-flash-lite" é o equivalente atual e mantém o tier gratuito.
GEMINI_MODEL = "gemini-3.5-flash-lite"
MODELOS_FALLBACK = ("gemini-3.5-flash-lite", "gemini-2.5-flash-lite")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"   # fallback quando a key não vem por parâmetro
GEMINI_API_KEY_VAR = "f/ofertas/gemini_api_key"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# "auto" | "requests" | "playwright"
ENGINE = "auto"

# Template exato pedido pelo negócio. `**` preservado de propósito; se preferir
# negrito nativo do WhatsApp (single-asterisk), troque por MENSAGEM_TEMPLATE_WA.
MENSAGEM_TEMPLATE = (
    "🔥 **{titulo}** 🔥\n\n"
    "⚡ *{gatilho}* ⚡\n\n"
    "💥 **Por apenas: R$ {preco} no PIX**\n\n"
    "🛒 **Garanta o seu aqui:** {link}\n\n"
    "⚠️ *Estoque limitado na cidade, corra para não ficar sem o seu drop!*"
)

# Variação com o markdown nativo do WhatsApp (*bold*, _itálico_)
MENSAGEM_TEMPLATE_WA = (
    "🔥 *{titulo}* 🔥\n\n"
    "⚡ _{gatilho}_ ⚡\n\n"
    "💥 *Por apenas: R$ {preco} no PIX*\n\n"
    "🛒 *Garanta o seu aqui:* {link}\n\n"
    "⚠️ _Estoque limitado na cidade, corra para não ficar sem o seu drop!_"
)

PALAVRAS_ESGOTADO = (
    "esgotado",
    "esgotada",
    "indisponivel",
    "indisponível",
    "sem estoque",
    "fora de estoque",
    "out of stock",
)

SYSTEM_PROMPT = (
    "Você é um copywriter de e-commerce especialista em vapes, pods descartáveis "
    "e e-liquids. Escreve em português do Brasil, curto, agressivo e sem exageros "
    "proibidos por política de anúncios (nada de promessa de saúde)."
)

PROMPT_COPY = """Título bruto do produto: "{titulo}"

Responda SOMENTE com um JSON válido, sem markdown, no formato:
{{"titulo_limpo": "...", "gatilho": "..."}}

Regras de "titulo_limpo":
- nome comercial curto e marcante, no máximo 42 caracteres
- mantenha marca + modelo + sabor principal (ex: "Ignite V400 Kiwi Ice")
- sem palavras genéricas ("produto", "refil", "novo"), sem preço, sem emoji

Regras de "gatilho":
- UMA frase curta de impacto sobre o sabor, no máximo 60 caracteres
- 1 ou 2 emojis no meio ou no fim
- fale de sensação/sabor (ex: "Explosão gelada de kiwi com maracujá 🥝❄️")"""


# ==========================================================================
# HTTP / RENDERIZAÇÃO
# ==========================================================================
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    return s


def _html_playwright(url: str, timeout_ms: int = 45000) -> str:
    """Headless Chrome com UA real. Levanta exceção se o Playwright não existir."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        try:
            ctx = browser.new_context(
                user_agent=USER_AGENT,
                locale="pt-BR",
                viewport={"width": 1366, "height": 900},
                extra_http_headers={"Accept-Language": "pt-BR,pt;q=0.9"},
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector("div.pcard", timeout=15000)
            except Exception:  # noqa: BLE001 - página sem grade de produtos
                pass
            # força o lazy-load das imagens antes de capturar o HTML
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(800)
            return page.content()
        finally:
            browser.close()


def buscar_html(url: str, engine: str = ENGINE, timeout: int = 45) -> str:
    """Retorna o HTML da URL: Playwright quando possível, senão requests."""
    if engine in ("playwright", "auto"):
        try:
            return _html_playwright(url)
        except Exception as e:  # noqa: BLE001
            if engine == "playwright":
                raise
            print(f"[scraper] Playwright indisponível ({e}); usando requests", flush=True)

    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# ==========================================================================
# LINK DE AFILIADO
# ==========================================================================
def converter_link_afiliado(url: str, ref: str = AFFILIATE_REF) -> str:
    """Absolutiza a URL do produto e injeta/atualiza `?ref=<ref>` sem duplicar."""
    if not url:
        return ""
    absoluta = urljoin(BASE_AFFILIATE_URL.rstrip("/") + "/", url.strip())
    partes = urlparse(absoluta)
    query = [(k, v) for k, v in parse_qsl(partes.query, keep_blank_values=True) if k.lower() != "ref"]
    query.append(("ref", ref))
    return urlunparse(partes._replace(query=urlencode(query), fragment=""))


# ==========================================================================
# EXTRAÇÃO DOS CARDS
# ==========================================================================
def _texto(el: Any) -> str:
    """Texto visível do elemento, ignorando comentários (marcadores de fragmento do Vue)."""
    if el is None:
        return ""
    partes = [
        s
        for s in el.find_all(string=True)
        if isinstance(s, NavigableString) and not isinstance(s, Comment)
    ]
    return re.sub(r"\s+", " ", " ".join(partes)).strip()


def _slug_titulo(url: str) -> str:
    caminho = urlparse(url).path.rstrip("/").split("/")[-1]
    return unicodedata.normalize("NFC", caminho.replace("-", " ")).title()


def _valor(txt: str) -> float | None:
    """'R$ 149,99' -> 149.99 ('1.149,90' também)."""
    m = re.search(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?", txt or "")
    if not m:
        return None
    inteiro = m.group(1).replace(".", "")
    centavos = (m.group(2) or "0").ljust(2, "0")
    try:
        return float(f"{inteiro}.{centavos}")
    except ValueError:
        return None


def _brl(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def extrair_produtos(
    html: str, page_url: str, apenas_disponiveis: bool = True
) -> list[dict]:
    """Lê os `div.pcard` da grade. Fallback: âncoras /produto/ quando não há grade."""
    soup = BeautifulSoup(html, "html.parser")
    produtos: list[dict] = []
    vistos: set[str] = set()

    for card in soup.select("div.pcard"):
        nome_el = card.select_one("a.pcard-name") or card.select_one(".pcard-name")
        titulo = _texto(nome_el)
        if not titulo:
            continue

        link_el = nome_el if (nome_el and nome_el.has_attr("href")) else card.select_one(
            "a[href*='/produto/']"
        )
        href = (link_el.get("href") if link_el else "") or ""
        if not href:
            continue
        link = urljoin(page_url, href)
        if link in vistos:
            continue

        card_txt = _texto(card).lower()
        tem_botao_compra = card.select_one("button.pcard-add, .pcard-add") is not None
        marcado_esgotado = any(p in card_txt for p in PALAVRAS_ESGOTADO)
        disponivel = tem_botao_compra and not marcado_esgotado
        if apenas_disponiveis and not disponivel:
            continue

        img_el = card.select_one("a.pcard-img img") or card.select_one("img")
        img = ""
        if img_el is not None:
            img = (
                img_el.get("data-src")
                or img_el.get("data-lazy-src")
                or img_el.get("src")
                or ""
            )
        imagem = urljoin(page_url, img) if img else ""

        pix_txt = _texto(card.select_one(".pcard-pix-val"))
        price_txt = _texto(card.select_one(".pcard-price"))
        preco_pix = _valor(pix_txt) or _valor(price_txt)
        preco_de = _valor(price_txt)

        produtos.append(
            {
                "id": (card.get("id") or "").replace("pcard-", ""),
                "titulo": titulo,
                "marca": _texto(card.select_one(".pcard-brand")),
                "url_produto": link,
                "url_da_imagem": imagem,
                "preco_pix": preco_pix,
                "preco_de": preco_de,
                "disponivel": disponivel,
            }
        )
        vistos.add(link)

    if produtos:
        return produtos

    # Fallback genérico (tema mudou / página sem grade)
    for a in soup.select("a[href*='/produto/']"):
        href = a.get("href") or ""
        link = urljoin(page_url, href)
        if not href or link in vistos:
            continue
        titulo = _texto(a) or _slug_titulo(link)
        if len(titulo) < 6:
            continue
        card = a.find_parent("div") or a
        img_el = card.select_one("img") if hasattr(card, "select_one") else None
        img = ""
        if img_el is not None:
            img = img_el.get("data-src") or img_el.get("src") or ""
        produtos.append(
            {
                "id": "",
                "titulo": titulo,
                "marca": "",
                "url_produto": link,
                "url_da_imagem": urljoin(page_url, img) if img else "",
                "preco_pix": _valor(_texto(card.select_one(".pcard-pix-val")) or _texto(card)),
                "preco_de": _valor(_texto(card.select_one(".pcard-price"))),
                "disponivel": not any(p in _texto(card).lower() for p in PALAVRAS_ESGOTADO),
            }
        )
        vistos.add(link)
    return produtos


def coletar_produtos(
    urls: list[str],
    max_paginas: int = 2,
    apenas_disponiveis: bool = True,
    engine: str = ENGINE,
) -> list[dict]:
    """Coleta 1..max_paginas de cada URL, deduplicando por URL de produto."""
    coletados: list[dict] = []
    vistos: set[str] = set()

    for url in urls:
        for pagina in range(1, max_paginas + 1):
            alvo = url if pagina == 1 else url.rstrip("/") + f"/page/{pagina}/"
            try:
                html = buscar_html(alvo, engine=engine)
            except Exception as e:  # noqa: BLE001
                print(f"[scraper] falha em {alvo}: {e}", flush=True)
                break

            lote = extrair_produtos(html, alvo, apenas_disponiveis=apenas_disponiveis)
            novos = [p for p in lote if p["url_produto"] not in vistos]
            print(f"[scraper] {alvo} -> {len(lote)} cards, {len(novos)} novos", flush=True)
            if not novos:
                break                                   # paginação acabou
            for p in novos:
                vistos.add(p["url_produto"])
                coletados.append(p)

    return coletados


# ==========================================================================
# GEMINI (google-genai)
# ==========================================================================
def _json_seguro(txt: str) -> dict:
    if not txt:
        return {}
    limpo = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", limpo, re.S)
    if not m:
        return {}
    try:
        dados = json.loads(m.group(0))
        return dados if isinstance(dados, dict) else {}
    except json.JSONDecodeError:
        return {}


def copy_fallback(titulo: str) -> dict:
    """Usado quando a API do Gemini falha: nunca deixa o card sem copy."""
    base = titulo.split(":")[0].strip() if len(titulo) > 42 else titulo
    base = re.sub(r"\s+", " ", base)[:42].strip().title()
    return {"titulo_limpo": base or titulo[:42], "gatilho": "Sabor insano, sensação premium 🔥💨"}


def gerar_copy_gemini(titulo: str, api_key: str, modelo: str = GEMINI_MODEL) -> dict:
    """Retorna {"titulo_limpo": ..., "gatilho": ...} validado."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resposta = client.models.generate_content(
        model=modelo,
        contents=PROMPT_COPY.format(titulo=titulo),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.9,
            response_mime_type="application/json",
            max_output_tokens=256,
        ),
    )
    dados = _json_seguro(getattr(resposta, "text", "") or "")
    titulo_limpo = str(dados.get("titulo_limpo") or "").strip()
    gatilho = str(dados.get("gatilho") or "").strip()
    if not titulo_limpo or not gatilho:
        raise ValueError(f"resposta do Gemini sem os campos esperados: {dados}")
    return {"titulo_limpo": titulo_limpo[:60], "gatilho": gatilho[:90]}


# Modelo que funcionou nesta execução: evita repetir o 404 a cada produto.
_MODELO_ATIVO: dict[str, str] = {}


def _cadeia_modelos(preferido: str) -> list[str]:
    cadeia: list[str] = []
    for m in (_MODELO_ATIVO.get("m"), preferido, *MODELOS_FALLBACK):
        if m and m not in cadeia:
            cadeia.append(m)
    return cadeia


def gerar_copy_com_fallback(titulo: str, api_key: str, preferido: str) -> dict:
    """Tenta o modelo pedido e, em 404/429, desce a cadeia MODELOS_FALLBACK."""
    erros: list[str] = []
    for modelo in _cadeia_modelos(preferido):
        try:
            copy = gerar_copy_gemini(titulo, api_key, modelo=modelo)
            _MODELO_ATIVO["m"] = modelo
            return copy
        except Exception as e:  # noqa: BLE001
            erros.append(f"{modelo}: {e}")
    raise RuntimeError(" | ".join(erros))


# ==========================================================================
# MENSAGEM FINAL
# ==========================================================================
def formatar_mensagem(titulo: str, gatilho: str, preco: str, link: str) -> str:
    return MENSAGEM_TEMPLATE.format(
        titulo=titulo.upper(), gatilho=gatilho, preco=preco, link=link
    )


def _evolution_payload(mensagem: str, imagem: str) -> dict:
    """Corpos prontos para o nó HTTP Request (Evolution API v2)."""
    return {
        "endpoint": "/message/sendMedia/{INSTANCIA}",
        "body": {
            "number": "{{NUMERO_DO_GRUPO}}",
            "mediatype": "image",
            "mimetype": "image/jpeg",
            "caption": mensagem,
            "media": imagem,
            "fileName": "oferta.jpg",
        },
        "endpoint_texto": "/message/sendText/{INSTANCIA}",
        "body_texto": {
            "number": "{{NUMERO_DO_GRUPO}}",
            "text": mensagem,
            "delay": 1200,
            "linkPreview": True,
        },
    }


# ==========================================================================
# MAIN
# ==========================================================================
def main(
    urls_alvo: list[str] | None = None,
    max_produtos: int = 3,
    max_paginas: int = 2,
    usar_gemini: bool = True,
    gemini_api_key: str = "",
    engine: str = "auto",
    apenas_disponiveis: bool = True,
    model: str = "gemini-3.5-flash-lite",
) -> list[dict]:
    """Retorna itens no formato do n8n: [{"json": {...}}, ...]"""
    alvos = [u for u in (urls_alvo or [TARGET_URL]) if u]

    api_key = (gemini_api_key or os.environ.get(GEMINI_API_KEY_ENV) or "").strip()
    if not api_key:
        try:
            import wmill

            api_key = (wmill.get_variable(GEMINI_API_KEY_VAR) or "").strip()
        except Exception:  # noqa: BLE001 - fora do Windmill ou variável inexistente
            api_key = ""

    produtos = coletar_produtos(
        alvos,
        max_paginas=max_paginas,
        apenas_disponiveis=apenas_disponiveis,
        engine=engine,
    )
    print(f"[main] {len(produtos)} produtos com estoque coletados", flush=True)

    itens: list[dict] = []
    sem_imagem = 0
    for produto in produtos:
        if len(itens) >= max_produtos:
            break
        if produto["preco_pix"] is None:
            print(f"[main] sem preço, ignorado: {produto['titulo']}", flush=True)
            continue

        titulo_bruto = produto["titulo"]
        if usar_gemini and api_key:
            try:
                copy = gerar_copy_com_fallback(titulo_bruto, api_key, model)
            except Exception as e:  # noqa: BLE001
                print(f"[gemini] fallback heurístico ({e}) para: {titulo_bruto}", flush=True)
                copy = copy_fallback(titulo_bruto)
        else:
            copy = copy_fallback(titulo_bruto)

        link = converter_link_afiliado(produto["url_produto"])
        preco = _brl(produto["preco_pix"])
        mensagem = formatar_mensagem(copy["titulo_limpo"], copy["gatilho"], preco, link)
        imagem = produto["url_da_imagem"] or produto["url_produto"]
        if not produto["url_da_imagem"]:
            sem_imagem += 1

        itens.append(
            {
                "json": {
                    "id": produto["id"],
                    "titulo_bruto": titulo_bruto,
                    "titulo_limpo": copy["titulo_limpo"],
                    "gatilho": copy["gatilho"],
                    "marca": produto["marca"],
                    "preco_pix": preco,
                    "preco_de": _brl(produto["preco_de"]),
                    "url_da_imagem": imagem,
                    "link_afiliado": link,
                    "mensagem_formatada_pelo_gemini": mensagem,
                    "evolution": _evolution_payload(mensagem, imagem),
                }
            }
        )

    print(
        f"[main] {len(itens)} itens prontos "
        f"(gemini_ativo={bool(usar_gemini and api_key)}, sem_imagem={sem_imagem})",
        flush=True,
    )
    return itens


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
