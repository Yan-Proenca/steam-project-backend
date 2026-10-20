import os
import json
import re
import functools
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

# Obtenha a URL do front-end a partir do ambiente ou use o padrão local para desenvolvimento
app = Flask(__name__)
CORS(app)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
PROFESSOR_MASTER_KEY = os.getenv("PROFESSOR_MASTER_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print(
        "⚠️  ALERTA: SUPABASE_URL / SUPABASE_ANON_KEY não configuradas. "
        "Elas são obrigatórias para TODAS as rotas de usuário final, pois é "
        "isso que faz o RLS ser aplicado por usuário. Pegue a 'anon/public "
        "key' em Supabase → Settings → API."
    )
if not SUPABASE_SERVICE_KEY:
    print(
        "⚠️  AVISO: SUPABASE_SERVICE_KEY não configurada — /api/auth/cadastro "
        "e a revogação de sessão em /api/auth/logout não vão funcionar."
    )

# -----------------------------------------------------------------------------
# Client administrativo (Service Role) — uso restrito, ver docstring acima.
# Estático (nunca sofre .auth(token) nem sign_in), então é seguro mantê-lo
# como singleton de módulo, compartilhado entre requisições.
# -----------------------------------------------------------------------------
supabase_admin: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None
)

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# =============================================================================
# HELPERS GERAIS
# =============================================================================
def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _uuid_valido(valor):
    """
    True se `valor` for None/vazio (campo opcional) OU um UUID bem formado.
    Usada para validar FKs opcionais (ex: sticker_recompensa_id) ANTES do
    insert/update, trocando um 500 cru do Postgres (22P02) por um 400
    amigável — geralmente sinal de que o front mandou o campo errado
    (ex: o `nome`/slug do sticker em vez do `id`).
    """
    if not valor:
        return True
    try:
        uuid.UUID(str(valor))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _parse_data_br(valor):
    """
    Normaliza uma data recebida do body para o formato ISO (AAAA-MM-DD)
    que o Postgres espera, aceitando dois formatos de entrada:
      - "DD/MM/AAAA"      (formato brasileiro — o que o front deve enviar)
      - "AAAA-MM-DD..."   (ISO, já pronto — aceito como fallback)
    Retorna None se `valor` for vazio/None (campo opcional).
    Levanta ValueError com mensagem amigável se o formato for irreconhecível
    ou a data não existir (ex: 31/02/2026).
    """
    if not valor:
        return None
    valor = str(valor).strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}", valor):
        return valor

    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", valor)
    if not m:
        raise ValueError("Data inválida. Use o formato DD/MM/AAAA.")

    dia, mes, ano = m.groups()
    try:
        datetime(int(ano), int(mes), int(dia))  # valida que a data existe de fato
    except ValueError:
        raise ValueError("Data inválida. Confira o dia e o mês informados.")
    return f"{ano}-{mes}-{dia}"


def _success(data=None, status=200):
    return jsonify({"success": True, "data": data}), status


def _error(message, status=400):
    return jsonify({"success": False, "error": message}), status


def _get_token_from_header():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


# =============================================================================
# VALIDAÇÃO DE E-MAIL E SENHA (cadastro/login)
# =============================================================================
_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# Mínimo 8 caracteres, 1 maiúscula, 1 minúscula, 1 número, 1 caractere especial
_SENHA_MAIUSCULA = re.compile(r"[A-Z]")
_SENHA_MINUSCULA = re.compile(r"[a-z]")
_SENHA_NUMERO = re.compile(r"[0-9]")
_SENHA_ESPECIAL = re.compile(r"[^A-Za-z0-9]")


def _normalizar_email(valor: str) -> str:
    """strip() + lower() — normalização padrão aplicada antes de validar/gravar."""
    return (valor or "").strip().lower()


def _validar_email(email: str):
    """
    Valida formato de e-mail via regex, já assumindo que `email` foi
    normalizado por `_normalizar_email` (strip + lower) antes.
    Retorna (True, None) se válido, ou (False, "mensagem de erro").
    """
    if not email:
        return False, "O e-mail é obrigatório."
    if not _EMAIL_REGEX.match(email):
        return False, "E-mail inválido. Verifique o formato digitado."
    return True, None


def _validar_senha(senha: str):
    """
    Exige mínimo de 8 caracteres, com pelo menos 1 letra maiúscula,
    1 letra minúscula, 1 número e 1 caractere especial.
    Retorna (True, None) se válida, ou (False, "mensagem de erro").
    """
    if not senha:
        return False, "A senha é obrigatória."
    if len(senha) < 8:
        return False, "A senha deve ter no mínimo 8 caracteres."
    if not _SENHA_MAIUSCULA.search(senha):
        return False, "A senha deve conter pelo menos 1 letra maiúscula."
    if not _SENHA_MINUSCULA.search(senha):
        return False, "A senha deve conter pelo menos 1 letra minúscula."
    if not _SENHA_NUMERO.search(senha):
        return False, "A senha deve conter pelo menos 1 número."
    if not _SENHA_ESPECIAL.search(senha):
        return False, "A senha deve conter pelo menos 1 caractere especial (ex: !@#$%&*)."
    return True, None


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detecta se uma exceção do SDK do Supabase Auth corresponde a rate limit
    de envio de e-mail (HTTP 429). O serviço de e-mail embutido do Supabase
    tem um limite muito baixo por padrão (2-4 e-mails/hora) — ver nota
    "CONFIGURAÇÃO DE SMTP" no topo do supabase_schema_completo.sql.
    """
    texto = str(exc).lower()
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    return status == 429 or "429" in texto or "rate limit" in texto or "too many requests" in texto


# =============================================================================
# UPLOAD DE ARQUIVOS — BUCKET ÚNICO "anexos_geral"
# =============================================================================
_ANEXOS_EXTENSOES_PERMITIDAS = {
    "pdf", "docx", "pptx", "xlsx", "zip",
    "png", "jpg", "jpeg", "txt", "py", "ipynb",
}
_ANEXOS_TAMANHO_MAXIMO_BYTES = 20 * 1024 * 1024  # 20 MB


def _upload_anexo_geral(db: Client, arquivo, prefixo: str):
    """
    Sobe `arquivo` (werkzeug FileStorage) para o bucket compartilhado
    "anexos_geral", sob o prefixo informado (ex: "missoes/", "entregas/"),
    com nome gerado via uuid.uuid4() para evitar colisões/sobrescritas.

    Retorna (url_publica, None) em caso de sucesso, ou (None, "mensagem de
    erro") em caso de falha de validação (extensão/tamanho) — nunca lança
    exceção por validação, só por falha de rede/Storage.
    """
    if not arquivo or not arquivo.filename:
        return None, "Nenhum arquivo selecionado."

    extensao = arquivo.filename.rsplit(".", 1)[-1].lower() if "." in arquivo.filename else ""
    if extensao not in _ANEXOS_EXTENSOES_PERMITIDAS:
        permitidas = ", ".join(sorted(_ANEXOS_EXTENSOES_PERMITIDAS))
        return None, f"Tipo de arquivo não permitido. Use um dos formatos: {permitidas}."

    conteudo = arquivo.read()
    if len(conteudo) > _ANEXOS_TAMANHO_MAXIMO_BYTES:
        return None, "Arquivo muito grande. O limite é 20 MB."

    # 🔒 Nome sempre gerado via uuid4() — nunca usa o nome original do
    # arquivo enviado pelo usuário, evitando colisões e path traversal.
    nome_arquivo = f"{prefixo}{uuid.uuid4()}.{extensao}"
    db.storage.from_("anexos_geral").upload(
        path=nome_arquivo,
        file=conteudo,
        file_options={"content-type": arquivo.content_type or "application/octet-stream"},
    )
    url = db.storage.from_("anexos_geral").get_public_url(nome_arquivo)
    return url, None


# =============================================================================
# CLIENTES SUPABASE POR REQUISIÇÃO (ANON KEY + JWT DO USUÁRIO)
# =============================================================================
def _new_anon_client() -> Client:
    """
    Cria uma instância NOVA do client Supabase com a ANON KEY, sem sessão.
    Usada para operações de auth públicas (sign_up / sign_in_with_password)
    e como base do client autenticado por requisição.

    IMPORTANTE: sempre uma instância nova, nunca um singleton global. O SDK
    do Supabase guarda a sessão de auth internamente no client e dispara
    `on_auth_state_change` a cada login, o que MUTA os headers desse client.
    Reutilizar um client global entre requisições concorrentes (gunicorn
    com múltiplas threads/workers) vazaria a sessão de um usuário para a
    requisição de outro usuário.
    """
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def get_user_supabase() -> Client:
    """
    Retorna um client Supabase (ANON KEY) autenticado com o Bearer JWT do
    usuário da requisição HTTP atual — para tabelas (PostgREST) e Storage.

    Sobrescrevemos `client.options.headers["Authorization"]` (mantendo
    "apikey" = anon key) ANTES de tocar em `.postgrest` ou `.storage`, pois
    ambos são inicializados de forma preguiçosa a partir desse mesmo dict de
    headers. Isso garante que TANTO as queries em tabelas QUANTO os uploads
    de arquivo respeitem o RLS do usuário logado — não só o Postgrest.
    `.postgrest.auth(token)` é chamado também, explicitamente, por clareza
    e redundância.

    Cacheado em `flask.g` (escopo de uma única requisição HTTP).
    """
    cached = getattr(g, "supabase_user_client", None)
    if cached is not None:
        return cached

    token = _get_token_from_header()
    if not token:
        raise PermissionError("Token de autenticação não fornecido.")

    client = _new_anon_client()
    client.options.headers["Authorization"] = f"Bearer {token}"
    client.postgrest.auth(token)

    g.supabase_user_client = client
    return client


# =============================================================================
# MIDDLEWARES DE AUTENTICAÇÃO / AUTORIZAÇÃO
# =============================================================================
def token_required(f):
    """
    Valida o JWT, monta o client Supabase escopado ao usuário (RLS) e
    injeta `current_user` (id, nome, role, email) na rota decorada.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _get_token_from_header()
        if not token:
            return _error("Token de autenticação não fornecido.", 401)
        try:
            db = get_user_supabase()

            user_resp = db.auth.get_user(token)
            auth_user = user_resp.user if user_resp else None
            if not auth_user:
                return _error("Token inválido ou expirado.", 401)

            perfil_r = (
                db.table("perfis")
                .select("id, nome, role, email")
                .eq("id", auth_user.id)
                .execute()
            )
            if not perfil_r.data:
                return _error("Perfil não encontrado.", 401)

            kwargs["current_user"] = perfil_r.data[0]
        except PermissionError as e:
            return _error(str(e), 401)
        except Exception as e:
            return _error(f"Falha na autenticação: {str(e)}", 401)
        return f(*args, **kwargs)
    return wrapper


def professor_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        current_user = kwargs.get("current_user")
        if not current_user or current_user.get("role") != "professor":
            return _error("Acesso restrito a professores.", 403)
        return f(*args, **kwargs)
    return wrapper


# =============================================================================
# VALIDAÇÃO DE POSSE (CAMADA 2 — checagem explícita além do RLS)
# -----------------------------------------------------------------------------
# Cada helper abaixo confirma que um recurso (período, missão, equipe,
# material) pertence à sala informada E que a sala pertence ao professor
# autenticado, retornando o registro ou None. As rotas transformam "None"
# em 404. Nunca fazemos `.eq("id", recurso_id)` isolado em PUT/DELETE.
# =============================================================================
def _sala_do_professor(db, sala_id, professor_id):
    r = (
        db.table("salas")
        .select("*")
        .eq("id", sala_id)
        .eq("professor_id", professor_id)
        .execute()
    )
    return r.data[0] if r.data else None


def _periodo_da_sala(db, periodo_id, sala_id, professor_id):
    if not _sala_do_professor(db, sala_id, professor_id):
        return None
    r = (
        db.table("periodos")
        .select("*")
        .eq("id", periodo_id)
        .eq("sala_id", sala_id)
        .execute()
    )
    return r.data[0] if r.data else None


def _missao_da_sala(db, missao_id, sala_id, professor_id):
    if not _sala_do_professor(db, sala_id, professor_id):
        return None
    r = (
        db.table("missoes")
        .select("*")
        .eq("id", missao_id)
        .eq("sala_id", sala_id)
        .execute()
    )
    return r.data[0] if r.data else None


def _equipe_da_sala(db, equipe_id, sala_id, professor_id):
    if not _sala_do_professor(db, sala_id, professor_id):
        return None
    r = (
        db.table("equipes")
        .select("*")
        .eq("id", equipe_id)
        .eq("sala_id", sala_id)
        .execute()
    )
    return r.data[0] if r.data else None


def _membro_da_equipe(db, equipe_id, aluno_id) -> bool:
    """
    Camada 2: confirma que `aluno_id` é de fato membro de `equipe_id`, além
    do RLS (policy equipe_tarefas_*/equipe_avisos_* já bloqueiam quem não é
    membro, mas repetimos a checagem aqui para devolver um 403 com mensagem
    clara em vez de um 404/sucesso vazio genérico do PostgREST).
    """
    r = (
        db.table("equipe_membros")
        .select("aluno_id")
        .eq("equipe_id", equipe_id)
        .eq("aluno_id", aluno_id)
        .execute()
    )
    return bool(r.data)


def _material_da_sala(db, material_id, sala_id, professor_id):
    if not _sala_do_professor(db, sala_id, professor_id):
        return None
    r = (
        db.table("biblioteca_materiais")
        .select("*")
        .eq("id", material_id)
        .eq("sala_id", sala_id)
        .execute()
    )
    return r.data[0] if r.data else None


def _progresso_da_sala_professor(db, progresso_id, professor_id):
    """
    Retorna o registro de progresso_missoes (com perfil do aluno e dados da
    missão embutidos) apenas se a missão associada pertencer a uma sala do
    professor autenticado. Usado por ver_entrega/corrigir_entrega para
    impedir que um professor veja ou corrija entregas de salas alheias.
    """
    prog_r = (
        db.table("progresso_missoes")
        .select(
            "*, perfis(id, nome, email), "
            "missoes(id, titulo, sala_id, peso_nota, periodo_id, sticker_recompensa_id)"
        )
        .eq("id", progresso_id)
        .execute()
    )
    if not prog_r.data:
        return None
    progresso = prog_r.data[0]
    missao = progresso.get("missoes")
    if not missao or not missao.get("sala_id"):
        return None
    if not _sala_do_professor(db, missao["sala_id"], professor_id):
        return None
    return progresso


# =============================================================================
# LÓGICA DE NEGÓCIO — TRILHA / GAMIFICAÇÃO (aluno)
# =============================================================================
def _get_vinculo_aluno(db, aluno_id: str):
    """Retorna o sala_id em que o aluno está matriculado, ou None."""
    r = db.table("aluno_salas").select("sala_id").eq("aluno_id", aluno_id).execute()
    return r.data[0]["sala_id"] if r.data else None


def _periodo_acessivel(db, aluno_id: str, sala_id: str, periodo_id: str) -> bool:
    """
    Um período só é acessível se todos os períodos anteriores (por ordem de
    criação) já tiverem 100% das missões PUBLICADAS validadas pelo professor.

    Implementação via RPC para a função SQL `_periodo_acessivel_sql`
    (supabase_schema_completo.sql, Seção 4) — mesma fonte de verdade usada
    por `missao_desbloqueada()`, para nunca mais divergir entre trilha,
    consulta e envio.
    """
    if not periodo_id:
        return True
    r = db.rpc("_periodo_acessivel_sql", {
        "p_aluno_id": aluno_id,
        "p_sala_id": sala_id,
        "p_periodo_id": periodo_id,
    }).execute()
    return bool(r.data)


def _missao_desbloqueada(db, aluno_id: str, missao_id: str) -> bool:
    """
    REGRA ÚNICA de desbloqueio na trilha — usada por TODAS as rotas que
    precisam decidir se uma missão está acessível para um aluno (trilha,
    consulta de uma missão, envio de entrega). Delega para a função SQL
    `missao_desbloqueada` (SECURITY DEFINER), que é a fonte de verdade
    única: publicada + todas as publicadas anteriores do mesmo período
    já corrigidas (validada_professor=TRUE) para este aluno.

    Antes desta unificação, cada uma dessas três rotas tinha sua própria
    lógica de desbloqueio, e elas divergiam entre si — esse era o bug
    relatado. Agora todas chamam este único helper.
    """
    r = db.rpc("missao_desbloqueada", {
        "p_aluno_id": aluno_id,
        "p_missao_id": missao_id,
    }).execute()
    return bool(r.data)


def _serializar_missao_para_aluno(missao: dict, desbloqueada: bool) -> dict:
    """
    Aplica a regra de rascunho/publicação na camada de aplicação (o RLS do
    banco permite ao aluno ler a LINHA inteira mesmo em rascunho — ver nota
    em supabase_schema_completo.sql, policy "missoes_select" — porque a
    decisão de produto foi: rascunho aparece bloqueado/cinza na trilha, só o
    CONTEÚDO fica oculto).

    Quando a missão está em rascunho, devolve apenas os campos estruturais
    (id, ordem, status) e omite todo o conteúdo (titulo real é substituído
    por um placeholder, descricao/o_que_entregar/anexo_url/xp_reward/
    sticker ficam None) — o suficiente para o front desenhar um card
    bloqueado/cinza na posição certa da trilha, sem revelar o que é a
    atividade.
    """
    if missao.get("status") == "rascunho":
        return {
            "id": missao["id"],
            "ordem": missao.get("ordem"),
            "periodo_id": missao.get("periodo_id"),
            "status": "rascunho",
            "titulo": "Em breve",
            "descricao": None,
            "o_que_entregar": None,
            "anexo_url": None,
            "xp_reward": None,
            "peso_nota": None,
            "sticker_recompensa_id": None,
            "stickers": None,
            "data_limite": None,
            "modalidade": missao.get("modalidade"),
            "status_aluno": "bloqueada",
            "desbloqueada": False,
            "progresso": None,
        }
    saida = dict(missao)
    saida["desbloqueada"] = desbloqueada
    return saida


def _build_periodos_trilha(db, aluno_id: str, sala_id: str):
    """
    Monta a trilha de missões agrupada por período, com status por missão.

    Usa `_missao_desbloqueada` (RPC → função SQL única) para decidir, missão
    a missão, se ela está acessível — a MESMA regra usada por
    `ver_missao_aluno` (consulta) e `enviar_missao` (envio), eliminando a
    divergência que existia antes entre essas três rotas.

    Missões em rascunho aparecem na trilha (ocupando sua posição/ordem,
    bloqueadas/cinza) mas sem conteúdo — ver `_serializar_missao_para_aluno`.
    Rascunhos NUNCA contam para o total/concluídas (não são atividades
    "de verdade" ainda) nem bloqueiam quem está avançando (a regra de
    desbloqueio, no banco, já ignora rascunhos na contagem de pendências).
    """
    periodos = (
        db.table("periodos")
        .select("id, nome")
        .eq("sala_id", sala_id)
        .order("criado_em")
        .execute().data
    )
    if not periodos:
        return [], 0, 0

    periodo_ids = [p["id"] for p in periodos]
    todas_missoes = (
        db.table("missoes")
        .select("*, stickers(imagem_url, nome, raridade)")
        .eq("sala_id", sala_id)
        .in_("periodo_id", periodo_ids)
        .order("ordem")
        .execute().data
    )

    ids_todas = [m["id"] for m in todas_missoes]
    progressos = (
        db.table("progresso_missoes")
        .select("*")
        .eq("aluno_id", aluno_id)
        .in_("missao_id", ids_todas)
        .execute().data
        if ids_todas else []
    )
    prog_idx = {p["missao_id"]: p for p in progressos}

    missoes_publicadas = [m for m in todas_missoes if m.get("status") == "publicada"]
    total = len(missoes_publicadas)
    concluidas = sum(
        1 for p in progressos
        if p.get("validada_professor") and p["missao_id"] in {m["id"] for m in missoes_publicadas}
    )

    periodos_trilha = []
    for periodo in periodos:
        missoes_periodo = [m for m in todas_missoes if m.get("periodo_id") == periodo["id"]]
        acessivel = _periodo_acessivel(db, aluno_id, sala_id, periodo["id"])

        missoes_out = []
        for m in missoes_periodo:
            prog = prog_idx.get(m["id"])
            desbloqueada = _missao_desbloqueada(db, aluno_id, m["id"])

            missao_serializada = _serializar_missao_para_aluno(m, desbloqueada)
            if m.get("status") == "publicada":
                if prog and prog.get("validada_professor"):
                    status_aluno = "concluida"
                elif not desbloqueada:
                    status_aluno = "bloqueada"
                elif prog and prog.get("status") == "corrigido":
                    status_aluno = "corrigida_sem_validacao"
                elif prog and prog.get("status") == "entregue":
                    status_aluno = "entregue"
                else:
                    status_aluno = "disponivel"
                missao_serializada["status_aluno"] = status_aluno
                missao_serializada["progresso"] = prog

            missoes_out.append(missao_serializada)

        periodos_trilha.append({**periodo, "missoes": missoes_out, "acessivel": acessivel})

    return periodos_trilha, total, concluidas


def _build_equipe_map_periodo(db, sala_id: str, periodo_id):
    """Retorna {aluno_id: nome_da_equipe} para o período informado (ou toda a sala)."""
    q = db.table("equipes").select("id, nome").eq("sala_id", sala_id)
    if periodo_id:
        q = q.eq("periodo_id", periodo_id)
    equipes = q.execute().data
    if not equipes:
        return {}

    equipe_ids = [e["id"] for e in equipes]
    nomes = {e["id"]: e["nome"] for e in equipes}
    membros = (
        db.table("equipe_membros")
        .select("equipe_id, aluno_id")
        .in_("equipe_id", equipe_ids)
        .execute().data
    )
    return {m["aluno_id"]: nomes.get(m["equipe_id"], "—") for m in membros}


def _equipe_do_aluno_no_periodo(db, aluno_id: str, sala_id: str, periodo_id):
    """
    Retorna a equipe (id, nome, membros) do aluno no período informado, ou
    None se ele não tem equipe nesse período. Usada para missões de
    modalidade='equipe': a decisão de produto é que o aluno PRECISA estar
    em uma equipe do período da missão para visualizar o conteúdo completo
    e enviar — sem equipe, ver_missao_aluno/enviar_missao bloqueiam com uma
    mensagem explicando o motivo (em vez de um 403 genérico).
    """
    q = db.table("equipes").select("id, nome").eq("sala_id", sala_id)
    q = q.eq("periodo_id", periodo_id) if periodo_id else q.is_("periodo_id", None)
    equipes = q.execute().data
    if not equipes:
        return None

    equipe_ids = [e["id"] for e in equipes]
    membro_r = (
        db.table("equipe_membros")
        .select("equipe_id")
        .eq("aluno_id", aluno_id)
        .in_("equipe_id", equipe_ids)
        .execute().data
    )
    if not membro_r:
        return None

    equipe_id = membro_r[0]["equipe_id"]
    equipe = next((e for e in equipes if e["id"] == equipe_id), None)
    if not equipe:
        return None

    membros_r = (
        db.table("equipe_membros")
        .select("perfis(id, nome)")
        .eq("equipe_id", equipe_id)
        .execute().data
    )
    membros = [m["perfis"] for m in membros_r if m.get("perfis")]
    return {"id": equipe["id"], "nome": equipe["nome"], "membros": membros}


# =============================================================================
# ROTA DE SAÚDE
# =============================================================================
@app.route("/")
def health():
    return _success({"status": "online", "service": "Plataforma Educacional Gamificada API"})


# =============================================================================
# AUTENTICAÇÃO
# =============================================================================
@app.route("/api/auth/cadastro", methods=["POST"])
def cadastro():
    """
    Cadastra um novo usuário (professor ou aluno).
    RN-G02: cadastro de professor exige a chave mestra.

    SEGURANÇA: `sign_up()` roda num client ANON KEY novo e descartável (é
    operação pública de auth). O upsert em `perfis` usa o client ADMIN
    (service_role) porque, quando a confirmação de e-mail está habilitada,
    `auth_response.session` vem None — ainda não existe um JWT do novo
    usuário para autenticar um client comum. Esta é a ÚNICA rota de negócio
    deste arquivo que usa a service_role key.
    """
    if supabase_admin is None:
        return _error("Serviço de cadastro indisponível (SUPABASE_SERVICE_KEY não configurada).", 500)

    body = request.get_json() or {}
    email = _normalizar_email(body.get("email", ""))
    password = body.get("senha", "")
    nome = body.get("nome", "").strip()
    role = body.get("role", "aluno")
    chave_mestra = body.get("chave_mestra", "")

    if not nome:
        return _error("Nome, e-mail e senha são obrigatórios.")

    email_ok, email_erro = _validar_email(email)
    if not email_ok:
        return _error(email_erro)

    senha_ok, senha_erro = _validar_senha(password)
    if not senha_ok:
        return _error(senha_erro)

    if role == "professor" and chave_mestra != PROFESSOR_MASTER_KEY:
        return _error("Chave mestra de professor incorreta.", 403)

    try:
        auth_client = _new_anon_client()
        auth_response = auth_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"nome": nome, "role": role}},
        })
        if not auth_response.user:
            return _error("Falha ao criar usuário no Supabase Auth.", 500)

        user = auth_response.user

        # 🔒 BUG CORRIGIDO: "cadastro repetido tentando criar perfil com ID
        # inexistente".
        #
        # O GoTrue (Auth do Supabase) NUNCA retorna erro de "e-mail já
        # cadastrado" em sign_up() — é proteção deliberada contra ataques de
        # enumeração de e-mail. Em vez disso, ele devolve um objeto de
        # sucesso "ofuscado" (HTTP 200, com user preenchido), e o único jeito
        # de distinguir isso de um cadastro novo de verdade é checar
        # `user.identities`:
        #
        #   • E-mail já existe E já está confirmado:
        #     GoTrue devolve um user FALSO (não existe em auth.users de
        #     verdade) com `identities == []`. O código antigo tentava
        #     upsert em perfis com esse id falso → violava a FK
        #     perfis.id → auth.users(id) → erro confuso de "ID inexistente".
        #     Agora detectamos isso ANTES de tentar o upsert e retornamos
        #     uma mensagem clara.
        #
        #   • E-mail já existe mas NÃO está confirmado ainda:
        #     GoTrue devolve o user REAL (identities preenchido), reenvia o
        #     e-mail de confirmação, e não gera sessão. Tratamos como
        #     "cadastro já existe, confirme seu e-mail" em vez de reportar
        #     sucesso de um cadastro novo.
        identidades = user.identities
        email_ja_existe_confirmado = identidades is not None and len(identidades) == 0

        if email_ja_existe_confirmado:
            return _error(
                "Este e-mail já está cadastrado. Faça login ou use 'Esqueci minha senha'.",
                409,
            )

        # E-mail já existe mas aguardando confirmação: como o GoTrue não
        # expõe isso de forma 100% distinguível de um cadastro novo em todas
        # as versões, usamos um sinal adicional e confiável — se já existe
        # uma linha em `perfis` para esse id (criada por um sign_up anterior
        # + trigger on_auth_user_created), este NÃO é um cadastro novo.
        perfil_preexistente = (
            supabase_admin.table("perfis").select("id").eq("id", user.id).execute()
        )
        if perfil_preexistente.data:
            return _error(
                "Este e-mail já está cadastrado e aguardando confirmação. "
                "Verifique sua caixa de entrada ou solicite um novo e-mail de confirmação.",
                409,
            )

        # Cadastro novo de verdade: usa upsert() para não colidir com o
        # trigger `on_auth_user_created` (SECURITY DEFINER), que também
        # popula `perfis` a partir dos metadados passados em options.data.
        supabase_admin.table("perfis").upsert({
            "id": user.id,
            "nome": nome,
            "role": role,
            "email": email,
        }, on_conflict="id").execute()

        return _success({"message": "Cadastro realizado! Verifique seu e-mail."}, 201)
    except Exception as e:
        if _is_rate_limit_error(e):
            return _error(
                "Muitas tentativas de cadastro em pouco tempo (limite de envio de e-mail do "
                "Supabase). Aguarde alguns minutos e tente novamente.",
                429,
            )
        return _error(f"Erro ao cadastrar: {str(e)}", 400)


@app.route("/api/auth/esqueci-senha", methods=["POST"])
def esqueci_senha():
    body = request.get_json() or {}
    email = _normalizar_email(body.get("email", ""))

    email_ok, email_erro = _validar_email(email)
    if not email_ok:
        return _error(email_erro)

    if supabase_admin is None:
        return _error("Serviço indisponível.", 500)

    try:
        # Gera o link de recuperação de forma administrativa sem disparar o e-mail nativo do Supabase
        response = supabase_admin.auth.admin.generate_link({
            "type": "recovery",
            "email": email
        })
        
        # O link gerado pelo Supabase para redefinição de senha
        action_link = response.properties.get("action_link") if response and response.properties else None
        
        if not action_link:
            # Por segurança, retornamos sucesso mesmo se o e-mail não existir para evitar enumeração
            return _success({
                "message": "Se este e-mail estiver cadastrado, as instruções foram geradas."
            })

        # Retornamos o link para o front-end disparar o EmailJS, ou você pode tratar aqui.
        # Nota: Por segurança em produção, idealmente o front-end chama o backend, 
        # o backend gera o link e o front-end usa o SDK do EmailJS com esse link.
        return _success({
            "message": "Link de recuperação gerado com sucesso.",
            "recovery_link": action_link,
            "email": email
        })

    except Exception as e:
        return _error(f"Erro ao processar solicitação: {str(e)}", 400)


@app.route("/api/auth/redefinir-senha", methods=["POST"])
@token_required
def redefinir_senha(current_user):
    """
    Redefine a senha do usuário autenticado. O "token" aqui é o access_token
    de recuperação que o Supabase emite ao clicar no link enviado pelo EmailJS.
    """
    body = request.get_json() or {}
    nova_senha = body.get("senha", "")

    senha_ok, senha_erro = _validar_senha(nova_senha)
    if not senha_ok:
        return _error(senha_erro)

    token = _get_token_from_header()
    try:
        # Mantém a mesma lógica robusta que já estava no seu app.py para atualizar a senha via token JWT de recuperação
        auth_client = _new_anon_client()
        auth_client.auth._request("PUT", "user", body={"password": nova_senha}, jwt=token)
        return _success({"message": "Senha redefinida com sucesso."})
    except Exception as e:
        return _error(f"Não foi possível redefinir a senha: {str(e)}", 400)


@app.route("/api/auth/login", methods=["POST"])
def login():
    """
    Autentica o usuário e retorna access_token + refresh_token + perfil.

    SEGURANÇA: `sign_in_with_password` roda num client ANON KEY novo e
    descartável (nunca um client global — evita vazar sessão entre
    requisições concorrentes). Assim que temos o `access_token` da sessão
    recém-criada, buscamos o perfil com um client autenticado NESSE token
    (não com a service key) — `perfis_select` permite a qualquer usuário
    autenticado ler perfis, então isso não expõe nada além do padrão já
    definido pelo RLS.
    """
    body = request.get_json() or {}
    email = _normalizar_email(body.get("email", ""))
    password = body.get("senha", "")

    email_ok, email_erro = _validar_email(email)
    if not email_ok:
        return _error(email_erro)
    if not password:
        return _error("A senha é obrigatória.")

    try:
        auth_client = _new_anon_client()
        response = auth_client.auth.sign_in_with_password({"email": email, "password": password})
        if not response.user or not response.session:
            return _error("Credenciais inválidas.", 401)

        access_token = response.session.access_token

        user_client = _new_anon_client()
        user_client.options.headers["Authorization"] = f"Bearer {access_token}"
        user_client.postgrest.auth(access_token)

        perfil_r = (
            user_client.table("perfis")
            .select("id, nome, role, email, xp")
            .eq("id", response.user.id)
            .execute()
        )
        perfil = perfil_r.data[0] if perfil_r.data else None

        return _success({
            "access_token": access_token,
            "refresh_token": response.session.refresh_token,
            "user": perfil,
        })
    except Exception as e:
        return _error(f"Erro ao fazer login: {str(e)}", 401)


@app.route("/api/auth/logout", methods=["POST"])
@token_required
def logout(current_user):
    """
    Revoga a sessão (refresh token) associada ao JWT atual.

    SEGURANÇA: `client.auth.sign_out()` do SDK depende de uma sessão
    guardada localmente pelo próprio SDK (via sign_in/set_session), que
    este backend stateless nunca mantém entre requisições. Por isso usamos
    a Admin API (`auth.admin.sign_out(jwt)`) para revogar explicitamente
    ESSE token específico. É uma operação administrativa sobre sessões de
    auth — não é leitura/escrita de dado de negócio — então não reintroduz
    o problema de bypass de RLS que este refactor resolve.
    """
    token = _get_token_from_header()
    try:
        if supabase_admin and token:
            supabase_admin.auth.admin.sign_out(token, scope="global")
    except Exception:
        pass
    return _success({"message": "Sessão encerrada."})


@app.route("/api/perfil", methods=["GET"])
@token_required
def ver_perfil(current_user):
    """
    Consulta atualizada do perfil do usuário autenticado: nome, e-mail e XP.

    Sempre busca direto na tabela `perfis` (não usa o `current_user` do
    decorator, que só tem id/nome/role/email) para trazer o `xp` já
    atualizado por qualquer trigger (ex: handle_mission_xp_reward) — o JWT
    em si não carrega XP, então este é o único jeito confiável de ter o
    valor corrente.

    NOTA: avatar/foto de perfil foi removido do produto; este endpoint
    retorna apenas nome, e-mail, role e xp.
    """
    db = get_user_supabase()
    r = (
        db.table("perfis")
        .select("id, nome, email, role, xp")
        .eq("id", current_user["id"])
        .execute()
    )
    if not r.data:
        return _error("Perfil não encontrado.", 404)
    return _success(r.data[0])


# =============================================================================
# PROFESSOR — SALAS
# =============================================================================
@app.route("/api/professor/salas", methods=["GET"])
@token_required
@professor_required
def listar_salas(current_user):
    db = get_user_supabase()
    professor_id = current_user["id"]
    salas = db.table("salas").select("*").eq("professor_id", professor_id).execute().data
    for sala in salas:
        sid = sala["id"]
        missoes_r = db.table("missoes").select("*", count="exact").eq("sala_id", sid).execute()
        alunos_r = db.table("aluno_salas").select("*", count="exact").eq("sala_id", sid).execute()
        sala["total_missoes"] = missoes_r.count or 0
        sala["total_alunos"] = alunos_r.count or 0
    return _success(salas)


@app.route("/api/professor/salas", methods=["POST"])
@token_required
@professor_required
def criar_sala(current_user):
    db = get_user_supabase()
    body = request.get_json() or {}
    nome_sala = body.get("nome", "").strip()
    if not nome_sala:
        return _error("O nome da sala é obrigatório.")
    try:
        codigo_sala = str(uuid.uuid4()).replace("-", "")[:6].upper()
        r = db.table("salas").insert({
            "nome": nome_sala,
            "professor_id": current_user["id"],
            "codigo_acesso": codigo_sala,
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>", methods=["GET"])
@token_required
@professor_required
def detalhes_sala(current_user, sala_id):
    db = get_user_supabase()
    sala = _sala_do_professor(db, sala_id, current_user["id"])
    if not sala:
        return _error("Sala não encontrada ou sem permissão.", 404)

    periodo_filtro = request.args.get("periodo_id")

    missoes = db.table("missoes").select("*").eq("sala_id", sala_id).order("ordem").execute().data
    periodos = db.table("periodos").select("id, nome").eq("sala_id", sala_id).execute().data

    equipes_q = db.table("equipes").select("*, periodos(nome)").eq("sala_id", sala_id)
    if periodo_filtro:
        equipes_q = equipes_q.eq("periodo_id", periodo_filtro)
    equipes = equipes_q.execute().data

    alunos_r = (
        db.table("aluno_salas")
        .select("perfis(id, nome, email)")
        .eq("sala_id", sala_id)
        .execute().data
    )
    alunos = [item["perfis"] for item in alunos_r if item.get("perfis")]

    return _success({
        "sala": sala,
        "missoes": missoes,
        "equipes": equipes,
        "alunos": alunos,
        "periodos": periodos,
    })


@app.route("/api/professor/salas/<sala_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_sala(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)
    try:
        missoes_r = db.table("missoes").select("id").eq("sala_id", sala_id).execute().data
        missao_ids = [m["id"] for m in missoes_r]

        equipes_r = db.table("equipes").select("id").eq("sala_id", sala_id).execute().data
        equipe_ids = [e["id"] for e in equipes_r]

        if equipe_ids:
            db.table("equipe_membros").delete().in_("equipe_id", equipe_ids).execute()
        db.table("equipes").delete().eq("sala_id", sala_id).execute()

        if missao_ids:
            db.table("progresso_missoes").delete().in_("missao_id", missao_ids).execute()
        db.table("missoes").delete().eq("sala_id", sala_id).execute()

        db.table("periodos").delete().eq("sala_id", sala_id).execute()
        db.table("aluno_salas").delete().eq("sala_id", sala_id).execute()
        db.table("salas").delete().eq("id", sala_id).execute()

        return _success({"message": "Sala excluída permanentemente."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — PERÍODOS (QUADRIMESTRES)
# =============================================================================
@app.route("/api/professor/salas/<sala_id>/periodos", methods=["GET"])
@token_required
@professor_required
def listar_periodos(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    periodos = (
        db.table("periodos")
        .select("id, nome, meta_missoes, criado_em")
        .eq("sala_id", sala_id)
        .order("criado_em")
        .execute().data
    )
    for p in periodos:
        cnt = (
            db.table("missoes")
            .select("*", count="exact")
            .eq("sala_id", sala_id)
            .eq("periodo_id", p["id"])
            .execute()
        )
        p["total_missoes"] = cnt.count or 0
    return _success(periodos)


@app.route("/api/professor/salas/<sala_id>/periodos", methods=["POST"])
@token_required
@professor_required
def criar_periodo(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    nome = body.get("nome", "").strip()
    if not nome:
        return _error("O nome do quadrimestre é obrigatório.")
    try:
        meta_missoes = max(1, min(15, int(body.get("meta_missoes", 5))))
    except (ValueError, TypeError):
        meta_missoes = 5

    try:
        r = db.table("periodos").insert({
            "sala_id": sala_id,
            "nome": nome,
            "meta_missoes": meta_missoes,
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/periodos/<periodo_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_periodo(current_user, sala_id, periodo_id):
    db = get_user_supabase()
    if not _periodo_da_sala(db, periodo_id, sala_id, current_user["id"]):
        return _error("Período não encontrado ou sem permissão.", 404)
    try:
        db.table("periodos").delete().eq("id", periodo_id).execute()
        return _success({"message": "Período removido."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — MISSÕES
# =============================================================================
def _ler_corpo_missao():
    """
    Lê os campos da missão tanto de JSON (Content-Type: application/json)
    quanto de multipart/form-data (quando um anexo é enviado junto).
    `request.files` só é populado em requisições multipart; nos demais
    casos, vem vazio — arquivo é sempre opcional.
    """
    if request.content_type and "multipart/form-data" in request.content_type:
        return request.form, request.files.get("anexo")
    return (request.get_json(silent=True) or {}), None


_STATUS_MISSAO_VALIDOS = {"rascunho", "publicada"}
_MODALIDADE_MISSAO_VALIDAS = {"individual", "equipe"}


def _ler_status_missao(body):
    """
    Lê e valida o campo "status" (rascunho/publicada) do corpo da
    requisição. Default: 'rascunho' — criar uma missão nunca a publica
    automaticamente (requisito: "salvar já disponibiliza a atividade" era o
    bug; agora é preciso publicar explicitamente).
    Retorna (status_valido, None) ou (None, "mensagem de erro").
    """
    status = (body.get("status") or "rascunho").strip().lower()
    if status not in _STATUS_MISSAO_VALIDOS:
        return None, f"Status inválido. Use um dos valores: {', '.join(sorted(_STATUS_MISSAO_VALIDOS))}."
    return status, None


def _ler_modalidade_missao(body):
    """
    Lê e valida o campo "modalidade" (individual/equipe). Default: 'individual'.
    Retorna (modalidade_valida, None) ou (None, "mensagem de erro").
    """
    modalidade = (body.get("modalidade") or "individual").strip().lower()
    if modalidade not in _MODALIDADE_MISSAO_VALIDAS:
        return None, f"Modalidade inválida. Use um dos valores: {', '.join(sorted(_MODALIDADE_MISSAO_VALIDAS))}."
    return modalidade, None


@app.route("/api/professor/salas/<sala_id>/missoes", methods=["POST"])
@token_required
@professor_required
def cadastrar_missao(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    body, anexo = _ler_corpo_missao()
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título da missão não pode estar vazio.")
    try:
        ordem = int(body.get("ordem", 1))
    except (ValueError, TypeError):
        return _error("Número de ordem inválido.")

    periodo_id = body.get("periodo_id") or None
    if periodo_id and not _periodo_da_sala(db, periodo_id, sala_id, current_user["id"]):
        return _error("Quadrimestre inválido para esta sala.", 404)

    q_ordem = db.table("missoes").select("id").eq("sala_id", sala_id).eq("ordem", ordem)
    q_ordem = q_ordem.eq("periodo_id", periodo_id) if periodo_id else q_ordem.is_("periodo_id", None)
    if q_ordem.execute().data:
        return _error(f"Conflito de Ordem: já existe uma missão com o número de ordem {ordem} neste período.", 400)

    if periodo_id:
        cnt = db.table("missoes").select("id", count="exact").eq("periodo_id", periodo_id).execute()
        if (cnt.count or 0) >= 5:
            return _error("Limite atingido: este período já possui o máximo de 5 missões.", 400)

    sticker_recompensa_id = body.get("sticker_recompensa_id") or None
    if not _uuid_valido(sticker_recompensa_id):
        return _error("Sticker inválido. Selecione um sticker da lista.", 400)

    try:
        data_limite = _parse_data_br(body.get("data_limite"))
    except ValueError as e:
        return _error(str(e), 400)

    status, erro_status = _ler_status_missao(body)
    if erro_status:
        return _error(erro_status, 400)

    modalidade, erro_modalidade = _ler_modalidade_missao(body)
    if erro_modalidade:
        return _error(erro_modalidade, 400)

    # 🔒 Anexo (opcional) vai para o bucket compartilhado "anexos_geral",
    # sob o prefixo "missoes/", com nome gerado via uuid4().
    anexo_url = None
    if anexo and anexo.filename:
        anexo_url, erro_anexo = _upload_anexo_geral(db, anexo, "missoes/")
        if erro_anexo:
            return _error(erro_anexo, 400)

    try:
        nova = {
            "sala_id": sala_id,
            "titulo": titulo,
            "descricao": body.get("descricao"),
            "o_que_entregar": body.get("o_que_entregar"),
            "ordem": ordem,
            "xp_reward": int(body.get("xp_reward", 0) or 0),
            "sticker_recompensa_id": sticker_recompensa_id,
            "data_limite": data_limite,
            "periodo_id": periodo_id,
            "peso_nota": float(body.get("peso_nota", 1) or 1),
            "anexo_url": anexo_url,
            "status": status,
            "modalidade": modalidade,
        }
        r = db.table("missoes").insert(nova).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/missoes/<missao_id>", methods=["PUT"])
@token_required
@professor_required
def editar_missao(current_user, sala_id, missao_id):
    db = get_user_supabase()
    missao_atual = _missao_da_sala(db, missao_id, sala_id, current_user["id"])
    if not missao_atual:
        return _error("Missão não encontrada ou sem permissão.", 404)

    body, anexo = _ler_corpo_missao()
    try:
        ordem = int(body.get("ordem", 1))
    except (ValueError, TypeError):
        return _error("Número de ordem inválido.")

    periodo_id = body.get("periodo_id") or None
    if periodo_id and not _periodo_da_sala(db, periodo_id, sala_id, current_user["id"]):
        return _error("Quadrimestre inválido para esta sala.", 404)

    q_ordem = (
        db.table("missoes")
        .select("id")
        .eq("sala_id", sala_id)
        .eq("ordem", ordem)
        .neq("id", missao_id)
    )
    q_ordem = q_ordem.eq("periodo_id", periodo_id) if periodo_id else q_ordem.is_("periodo_id", None)
    if q_ordem.execute().data:
        return _error(f"Conflito de Ordem: o número {ordem} já está em uso por outra missão.", 400)

    if periodo_id and missao_atual.get("periodo_id") != periodo_id:
        cnt = db.table("missoes").select("id", count="exact").eq("periodo_id", periodo_id).execute()
        if (cnt.count or 0) >= 5:
            return _error("Não é possível mover esta missão: o período de destino já tem 5 missões.", 400)

    sticker_recompensa_id = body.get("sticker_recompensa_id") or None
    if not _uuid_valido(sticker_recompensa_id):
        return _error("Sticker inválido. Selecione um sticker da lista.", 400)

    try:
        data_limite = _parse_data_br(body.get("data_limite"))
    except ValueError as e:
        return _error(str(e), 400)

    status, erro_status = _ler_status_missao(body)
    if erro_status:
        return _error(erro_status, 400)

    modalidade, erro_modalidade = _ler_modalidade_missao(body)
    if erro_modalidade:
        return _error(erro_modalidade, 400)

    # 🔒 Anexo (opcional) substitui o anterior; se nada for enviado, o
    # campo "anexo_url" não é alterado (missão mantém o anexo existente).
    dados = {
        "titulo": body.get("titulo"),
        "descricao": body.get("descricao"),
        "o_que_entregar": body.get("o_que_entregar"),
        "ordem": ordem,
        "xp_reward": int(body.get("xp_reward", 0) or 0),
        "sticker_recompensa_id": sticker_recompensa_id,
        "data_limite": data_limite,
        "periodo_id": periodo_id,
        "peso_nota": float(body.get("peso_nota", 1) or 1),
        "status": status,
        "modalidade": modalidade,
    }
    if anexo and anexo.filename:
        anexo_url, erro_anexo = _upload_anexo_geral(db, anexo, "missoes/")
        if erro_anexo:
            return _error(erro_anexo, 400)
        dados["anexo_url"] = anexo_url

    try:
        r = db.table("missoes").update(dados).eq("id", missao_id).execute()
        return _success(r.data[0] if r.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/missoes/<missao_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_missao(current_user, sala_id, missao_id):
    db = get_user_supabase()
    if not _missao_da_sala(db, missao_id, sala_id, current_user["id"]):
        return _error("Missão não encontrada ou sem permissão.", 404)
    try:
        db.table("missoes").delete().eq("id", missao_id).execute()
        return _success({"message": "Missão removida da trilha."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# STICKERS — SEM CRUD NO BACKEND
# -----------------------------------------------------------------------------
# O catálogo de stickers (public.stickers) e as imagens no bucket "stickers"
# são geridos EXCLUSIVAMENTE pelo Admin via Supabase Dashboard (Table Editor
# + Storage), usando a service_role key. Esta API não expõe mais nenhuma
# rota de CRUD para stickers — eles continuam sendo lidos (somente leitura,
# via RLS `stickers_select`) e vinculados a missões/quizzes como recompensa,
# conforme já implementado em cadastrar_missao/editar_missao, corrigir_entrega
# e handle_quiz_completion_rewards.
# =============================================================================


# =============================================================================
# PROFESSOR — ALUNOS DA SALA
# =============================================================================
@app.route("/api/professor/salas/<sala_id>/alunos", methods=["POST"])
@token_required
@professor_required
def adicionar_aluno_sala(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    email_aluno = body.get("email", "").strip().lower()
    if not email_aluno:
        return _error("Informe o e-mail do aluno.")

    try:
        aluno_r = (
            db.table("perfis")
            .select("id, nome")
            .eq("email", email_aluno)
            .eq("role", "aluno")
            .execute()
        )
        if not aluno_r.data:
            return _error("Aluno não encontrado ou não possui conta de aluno.", 404)
        aluno = aluno_r.data[0]

        ja_tem = db.table("aluno_salas").select("sala_id").eq("aluno_id", aluno["id"]).execute()
        if ja_tem.data:
            return _error(f"O aluno '{aluno['nome']}' já está matriculado em outra sala.", 409)

        db.table("aluno_salas").insert({"aluno_id": aluno["id"], "sala_id": sala_id}).execute()
        return _success({"message": f"{aluno['nome']} adicionado à sala!"}, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/alunos/<aluno_id>", methods=["DELETE"])
@token_required
@professor_required
def remover_aluno_sala(current_user, sala_id, aluno_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)
    try:
        db.table("aluno_salas").delete().eq("sala_id", sala_id).eq("aluno_id", aluno_id).execute()
        return _success({"message": "Aluno removido da sala."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — EQUIPES
# =============================================================================
@app.route("/api/professor/salas/<sala_id>/equipes", methods=["POST"])
@token_required
@professor_required
def criar_equipe(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    nome = body.get("nome", "").strip()
    periodo_id = body.get("periodo_id", "").strip() if body.get("periodo_id") else ""
    if not nome:
        return _error("O nome da equipe não pode ser vazio.")
    if not periodo_id:
        return _error("Selecione um quadrimestre para criar a equipe.")
    if not _periodo_da_sala(db, periodo_id, sala_id, current_user["id"]):
        return _error("Quadrimestre inválido para esta sala.", 404)

    try:
        r = db.table("equipes").insert({
            "nome": nome, "sala_id": sala_id, "periodo_id": periodo_id,
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>", methods=["GET"])
@token_required
@professor_required
def detalhes_equipe(current_user, sala_id, equipe_id):
    db = get_user_supabase()
    equipe = _equipe_da_sala(db, equipe_id, sala_id, current_user["id"])
    if not equipe:
        return _error("Equipe não encontrada ou sem permissão.", 404)

    equipe_r = db.table("equipes").select("*, periodos(nome)").eq("id", equipe_id).execute()
    if equipe_r.data:
        equipe = equipe_r.data[0]

    membros_r = db.table("equipe_membros").select("perfis(id, nome, email)").eq("equipe_id", equipe_id).execute().data
    membros = [m["perfis"] for m in membros_r if m.get("perfis")]

    todos_r = db.table("aluno_salas").select("perfis(id, nome)").eq("sala_id", sala_id).execute().data
    todos_alunos = [item["perfis"] for item in todos_r if item.get("perfis")]

    periodo_id_equipe = equipe.get("periodo_id")
    q = db.table("equipes").select("id").eq("sala_id", sala_id)
    if periodo_id_equipe:
        q = q.eq("periodo_id", periodo_id_equipe)
    equipes_mesmo_periodo = q.execute().data
    ids_equipes_periodo = [e["id"] for e in equipes_mesmo_periodo]

    ids_em_equipe_periodo = set()
    if ids_equipes_periodo:
        m_r = db.table("equipe_membros").select("aluno_id").in_("equipe_id", ids_equipes_periodo).execute().data
        ids_em_equipe_periodo = {m["aluno_id"] for m in m_r}

    alunos_disponiveis = [a for a in todos_alunos if a["id"] not in ids_em_equipe_periodo]

    return _success({
        "equipe": equipe,
        "membros": membros,
        "alunos_disponiveis": alunos_disponiveis,
    })


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_equipe(current_user, sala_id, equipe_id):
    db = get_user_supabase()
    if not _equipe_da_sala(db, equipe_id, sala_id, current_user["id"]):
        return _error("Equipe não encontrada ou sem permissão.", 404)
    try:
        db.table("equipe_membros").delete().eq("equipe_id", equipe_id).execute()
        db.table("equipes").delete().eq("id", equipe_id).execute()
        return _success({"message": "Equipe removida."})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/membros", methods=["POST"])
@token_required
@professor_required
def adicionar_membro_equipe(current_user, sala_id, equipe_id):
    db = get_user_supabase()
    equipe = _equipe_da_sala(db, equipe_id, sala_id, current_user["id"])
    if not equipe:
        return _error("Equipe não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    aluno_id = body.get("aluno_id")
    if not aluno_id:
        return _error("Informe o aluno.")

    # 🔒 Garante que o aluno pertence à MESMA sala da equipe (não apenas
    # que ele existe em algum lugar da plataforma).
    matricula = (
        db.table("aluno_salas")
        .select("aluno_id")
        .eq("sala_id", sala_id)
        .eq("aluno_id", aluno_id)
        .execute()
    )
    if not matricula.data:
        return _error("Este aluno não está matriculado nesta sala.", 400)

    try:
        periodo_id_equipe = equipe.get("periodo_id")
        if periodo_id_equipe:
            equipes_periodo = (
                db.table("equipes")
                .select("id")
                .eq("sala_id", sala_id)
                .eq("periodo_id", periodo_id_equipe)
                .execute().data
            )
            ids_periodo = [e["id"] for e in equipes_periodo]
            if ids_periodo:
                ja_em_equipe = (
                    db.table("equipe_membros")
                    .select("equipe_id")
                    .eq("aluno_id", aluno_id)
                    .in_("equipe_id", ids_periodo)
                    .execute().data
                )
                if ja_em_equipe:
                    return _error("Este aluno já pertence a uma equipe neste quadrimestre.", 409)

        db.table("equipe_membros").insert({"equipe_id": equipe_id, "aluno_id": aluno_id}).execute()
        return _success({"message": "Aluno adicionado à equipe!"}, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/membros/<aluno_id>", methods=["DELETE"])
@token_required
@professor_required
def remover_membro_equipe(current_user, sala_id, equipe_id, aluno_id):
    db = get_user_supabase()
    if not _equipe_da_sala(db, equipe_id, sala_id, current_user["id"]):
        return _error("Equipe não encontrada ou sem permissão.", 404)
    try:
        db.table("equipe_membros").delete().eq("equipe_id", equipe_id).eq("aluno_id", aluno_id).execute()
        return _success({"message": "Aluno removido da equipe."})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/tarefas", methods=["GET"])
@token_required
@professor_required
def tarefas_equipe_professor(current_user, sala_id, equipe_id):
    """Professor da sala SÓ VISUALIZA as tarefas da equipe (decisão de produto)."""
    db = get_user_supabase()
    if not _equipe_da_sala(db, equipe_id, sala_id, current_user["id"]):
        return _error("Equipe não encontrada ou sem permissão.", 404)
    tarefas = (
        db.table("equipe_tarefas")
        .select("*, responsavel:responsavel_id(id, nome)")
        .eq("equipe_id", equipe_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    return _success({"tarefas": tarefas})


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/avisos", methods=["GET"])
@token_required
@professor_required
def avisos_equipe_professor(current_user, sala_id, equipe_id):
    """Professor da sala SÓ VISUALIZA o mural de avisos da equipe (decisão de produto)."""
    db = get_user_supabase()
    if not _equipe_da_sala(db, equipe_id, sala_id, current_user["id"]):
        return _error("Equipe não encontrada ou sem permissão.", 404)
    avisos = (
        db.table("equipe_avisos")
        .select("*, autor:autor_id(id, nome)")
        .eq("equipe_id", equipe_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    return _success({"avisos": avisos})


# =============================================================================
# ALUNO — EQUIPE (tarefas compartilhadas + mural de avisos)
# -----------------------------------------------------------------------------
# Cadastro de equipes e associação de integrantes são geridos pelo professor
# (seção acima). Aqui: o próprio aluno consultando sua equipe, e os membros
# gerenciando tarefas/avisos — exclusivamente entre eles (o professor só lê,
# nas rotas acima).
# =============================================================================
@app.route("/api/aluno/equipe", methods=["GET"])
@token_required
def minha_equipe(current_user):
    """
    Equipe do aluno no período CORRENTE da sua sala (o período mais recente
    por data de criação — mesmo critério usado na trilha). Se o aluno não
    tem sala, ou não tem equipe nesse período, devolve equipe=null (não é
    erro: nem toda missão é de equipe).
    """
    db = get_user_supabase()
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(db, aluno_id)
    if not sala_id:
        return _success({"equipe": None})

    periodo_atual = (
        db.table("periodos")
        .select("id")
        .eq("sala_id", sala_id)
        .order("criado_em", desc=True)
        .limit(1)
        .execute().data
    )
    periodo_id = periodo_atual[0]["id"] if periodo_atual else None

    equipe = _equipe_do_aluno_no_periodo(db, aluno_id, sala_id, periodo_id)
    return _success({"equipe": equipe})


def _validar_membro_equipe_aluno(db, equipe_id, aluno_id):
    """Helper comum às rotas de tarefas/avisos do aluno: aluno precisa ser membro da equipe."""
    if not _membro_da_equipe(db, equipe_id, aluno_id):
        return _error("Você não é membro desta equipe.", 403)
    return None


@app.route("/api/aluno/equipes/<equipe_id>/tarefas", methods=["GET"])
@token_required
def listar_tarefas_equipe(current_user, equipe_id):
    db = get_user_supabase()
    erro = _validar_membro_equipe_aluno(db, equipe_id, current_user["id"])
    if erro:
        return erro
    tarefas = (
        db.table("equipe_tarefas")
        .select("*, responsavel:responsavel_id(id, nome)")
        .eq("equipe_id", equipe_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    return _success({"tarefas": tarefas})


@app.route("/api/aluno/equipes/<equipe_id>/tarefas", methods=["POST"])
@token_required
def criar_tarefa_equipe(current_user, equipe_id):
    """Qualquer MEMBRO da equipe pode criar uma tarefa (decisão de produto: sem hierarquia interna)."""
    db = get_user_supabase()
    aluno_id = current_user["id"]
    erro = _validar_membro_equipe_aluno(db, equipe_id, aluno_id)
    if erro:
        return erro

    body = request.get_json() or {}
    titulo = (body.get("titulo") or "").strip()
    if not titulo:
        return _error("O título da tarefa é obrigatório.")

    responsavel_id = body.get("responsavel_id") or None
    if responsavel_id and not _membro_da_equipe(db, equipe_id, responsavel_id):
        return _error("O responsável precisa ser um membro desta equipe.", 400)

    try:
        prazo = _parse_data_br(body.get("prazo"))
    except ValueError as e:
        return _error(str(e), 400)

    status = (body.get("status") or "pendente").strip().lower()
    if status not in ("pendente", "em_andamento", "concluida"):
        return _error("Status inválido. Use 'pendente', 'em_andamento' ou 'concluida'.", 400)

    try:
        # 🔒 criado_por vem sempre de current_user, nunca do body.
        r = db.table("equipe_tarefas").insert({
            "equipe_id": equipe_id,
            "titulo": titulo,
            "responsavel_id": responsavel_id,
            "prazo": prazo,
            "status": status,
            "criado_por": aluno_id,
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


def _tarefa_da_equipe_do_aluno(db, tarefa_id, aluno_id):
    """Confirma que a tarefa existe e que o aluno é membro da equipe dona dela."""
    r = db.table("equipe_tarefas").select("*").eq("id", tarefa_id).execute()
    if not r.data:
        return None
    tarefa = r.data[0]
    if not _membro_da_equipe(db, tarefa["equipe_id"], aluno_id):
        return None
    return tarefa


@app.route("/api/aluno/tarefas/<tarefa_id>", methods=["PUT"])
@token_required
def editar_tarefa_equipe(current_user, tarefa_id):
    """Edita título/responsável/prazo/status. Qualquer membro da equipe pode editar (inclui concluir)."""
    db = get_user_supabase()
    tarefa = _tarefa_da_equipe_do_aluno(db, tarefa_id, current_user["id"])
    if not tarefa:
        return _error("Tarefa não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    titulo = (body.get("titulo") or tarefa["titulo"]).strip()
    if not titulo:
        return _error("O título da tarefa é obrigatório.")

    responsavel_id = body.get("responsavel_id", tarefa.get("responsavel_id"))
    if responsavel_id and not _membro_da_equipe(db, tarefa["equipe_id"], responsavel_id):
        return _error("O responsável precisa ser um membro desta equipe.", 400)

    status = (body.get("status") or tarefa["status"]).strip().lower()
    if status not in ("pendente", "em_andamento", "concluida"):
        return _error("Status inválido. Use 'pendente', 'em_andamento' ou 'concluida'.", 400)

    try:
        prazo = _parse_data_br(body.get("prazo")) if "prazo" in body else tarefa.get("prazo")
    except ValueError as e:
        return _error(str(e), 400)

    try:
        r = (
            db.table("equipe_tarefas")
            .update({
                "titulo": titulo,
                "responsavel_id": responsavel_id,
                "prazo": prazo,
                "status": status,
                "atualizado_em": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", tarefa_id)
            .execute()
        )
        return _success(r.data[0] if r.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/tarefas/<tarefa_id>", methods=["DELETE"])
@token_required
def excluir_tarefa_equipe(current_user, tarefa_id):
    db = get_user_supabase()
    tarefa = _tarefa_da_equipe_do_aluno(db, tarefa_id, current_user["id"])
    if not tarefa:
        return _error("Tarefa não encontrada ou sem permissão.", 404)
    try:
        db.table("equipe_tarefas").delete().eq("id", tarefa_id).execute()
        return _success({"message": "Tarefa removida."})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/equipes/<equipe_id>/avisos", methods=["GET"])
@token_required
def listar_avisos_equipe(current_user, equipe_id):
    db = get_user_supabase()
    erro = _validar_membro_equipe_aluno(db, equipe_id, current_user["id"])
    if erro:
        return erro
    avisos = (
        db.table("equipe_avisos")
        .select("*, autor:autor_id(id, nome)")
        .eq("equipe_id", equipe_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    return _success({"avisos": avisos})


@app.route("/api/aluno/equipes/<equipe_id>/avisos", methods=["POST"])
@token_required
def criar_aviso_equipe(current_user, equipe_id):
    """Qualquer membro da equipe pode publicar um aviso no mural."""
    db = get_user_supabase()
    aluno_id = current_user["id"]
    erro = _validar_membro_equipe_aluno(db, equipe_id, aluno_id)
    if erro:
        return erro

    body = request.get_json() or {}
    texto = (body.get("texto") or "").strip()
    if not texto:
        return _error("O texto do aviso é obrigatório.")

    try:
        # 🔒 autor_id vem sempre de current_user, nunca do body.
        r = db.table("equipe_avisos").insert({
            "equipe_id": equipe_id,
            "autor_id": aluno_id,
            "texto": texto,
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/avisos/<aviso_id>", methods=["PUT"])
@token_required
def editar_aviso_equipe(current_user, aviso_id):
    """Só o AUTOR do aviso pode editá-lo (decisão de produto)."""
    db = get_user_supabase()
    aluno_id = current_user["id"]

    r = db.table("equipe_avisos").select("*").eq("id", aviso_id).execute()
    if not r.data:
        return _error("Aviso não encontrado.", 404)
    aviso = r.data[0]
    if aviso["autor_id"] != aluno_id:
        return _error("Você só pode editar os próprios avisos.", 403)

    body = request.get_json() or {}
    texto = (body.get("texto") or "").strip()
    if not texto:
        return _error("O texto do aviso é obrigatório.")

    try:
        upd = (
            db.table("equipe_avisos")
            .update({"texto": texto, "atualizado_em": datetime.now(timezone.utc).isoformat()})
            .eq("id", aviso_id)
            .execute()
        )
        return _success(upd.data[0] if upd.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/avisos/<aviso_id>", methods=["DELETE"])
@token_required
def excluir_aviso_equipe(current_user, aviso_id):
    """Só o AUTOR do aviso pode excluí-lo (decisão de produto)."""
    db = get_user_supabase()
    aluno_id = current_user["id"]

    r = db.table("equipe_avisos").select("autor_id").eq("id", aviso_id).execute()
    if not r.data:
        return _error("Aviso não encontrado.", 404)
    if r.data[0]["autor_id"] != aluno_id:
        return _error("Você só pode excluir os próprios avisos.", 403)

    try:
        db.table("equipe_avisos").delete().eq("id", aviso_id).execute()
        return _success({"message": "Aviso removido."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — ENTREGAS / CORREÇÃO
# =============================================================================
@app.route("/api/professor/salas/<sala_id>/entregas", methods=["GET"])
@token_required
@professor_required
def listar_entregas(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    filtro_missao = request.args.get("missao_id", "").strip()
    filtro_equipe = request.args.get("equipe_id", "").strip()
    filtro_periodo = request.args.get("periodo_id", "").strip()

    missoes_q = db.table("missoes").select("id, titulo, data_limite, periodo_id").eq("sala_id", sala_id)
    if filtro_periodo:
        missoes_q = missoes_q.eq("periodo_id", filtro_periodo)
    if filtro_missao:
        missoes_q = missoes_q.eq("id", filtro_missao)
    missoes_r = missoes_q.order("ordem").execute().data

    todas_missoes_r = db.table("missoes").select("id, titulo, periodo_id").eq("sala_id", sala_id).order("ordem").execute().data
    periodos_r = db.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data
    equipes_r = db.table("equipes").select("id, nome, periodo_id").eq("sala_id", sala_id).order("nome").execute().data

    equipe_ids = [e["id"] for e in equipes_r]
    equipes_obj = {e["id"]: e for e in equipes_r}

    aluno_equipe_por_periodo = {}
    if equipe_ids:
        membros_r = db.table("equipe_membros").select("equipe_id, aluno_id").in_("equipe_id", equipe_ids).execute().data
        for m in membros_r:
            eq = equipes_obj.get(m["equipe_id"])
            if eq and eq.get("periodo_id"):
                aluno_equipe_por_periodo[(m["aluno_id"], eq["periodo_id"])] = eq

    missao_ids = [m["id"] for m in missoes_r]
    missoes_map = {m["id"]: m["titulo"] for m in todas_missoes_r}
    missao_periodo_map = {m["id"]: m.get("periodo_id") for m in todas_missoes_r}
    deadline_map = {m["id"]: m.get("data_limite") for m in todas_missoes_r}

    entregas = []
    if missao_ids:
        progressos = (
            db.table("progresso_missoes")
            .select("*, perfis(id, nome, email)")
            .in_("missao_id", missao_ids)
            .neq("status", "pendente")
            .order("entregue_em", desc=True)
            .execute().data
        )

        if filtro_equipe:
            membros_da_equipe = {
                m["aluno_id"] for m in
                db.table("equipe_membros").select("aluno_id").eq("equipe_id", filtro_equipe).execute().data
            }
            progressos = [p for p in progressos if p["aluno_id"] in membros_da_equipe]

        for p in progressos:
            periodo_da_miss = missao_periodo_map.get(p["missao_id"])
            equipe_obj = aluno_equipe_por_periodo.get((p["aluno_id"], periodo_da_miss))
            entregas.append({
                **p,
                "titulo_missao": missoes_map.get(p["missao_id"], "—"),
                "nome_equipe": equipe_obj["nome"] if equipe_obj else "Sem equipe",
                "data_limite": deadline_map.get(p["missao_id"]),
            })

    return _success({
        "entregas": entregas,
        "missoes": todas_missoes_r,
        "periodos": periodos_r,
        "equipes": equipes_r,
        "resumo": {
            "total": len(entregas),
            "pendentes": sum(1 for e in entregas if e["status"] == "entregue"),
            "corrigidas": sum(1 for e in entregas if e["status"] == "corrigido"),
        },
    })


@app.route("/api/professor/entregas/<progresso_id>", methods=["GET"])
@token_required
@professor_required
def ver_entrega(current_user, progresso_id):
    db = get_user_supabase()
    progresso = _progresso_da_sala_professor(db, progresso_id, current_user["id"])
    if not progresso:
        return _error("Entrega não encontrada ou sem permissão.", 404)

    sala_id = progresso["missoes"]["sala_id"]
    periodo_id_missao = progresso["missoes"].get("periodo_id")

    equipe = None
    membros_equipe = []
    if periodo_id_missao:
        equipes_r = (
            db.table("equipes")
            .select("id, nome")
            .eq("sala_id", sala_id)
            .eq("periodo_id", periodo_id_missao)
            .execute().data
        )
        ids_equipes = [e["id"] for e in equipes_r]
        equipes_map = {e["id"]: e for e in equipes_r}
        if ids_equipes:
            membro_r = (
                db.table("equipe_membros")
                .select("equipe_id")
                .eq("aluno_id", progresso["aluno_id"])
                .in_("equipe_id", ids_equipes)
                .execute().data
            )
            if membro_r:
                equipe_id = membro_r[0]["equipe_id"]
                equipe = equipes_map.get(equipe_id)
                membros_r = db.table("equipe_membros").select("perfis(nome)").eq("equipe_id", equipe_id).execute().data
                membros_equipe = [m["perfis"]["nome"] for m in membros_r if m.get("perfis")]

    anexos = (
        db.table("progresso_missoes_anexos")
        .select("id, url, nome_arquivo, criado_em")
        .eq("progresso_id", progresso_id)
        .order("criado_em")
        .execute().data
    )

    return _success({
        "progresso": progresso,
        "anexos": anexos,
        "equipe": equipe,
        "membros": membros_equipe,
        "sala_id": sala_id,
    })


def _conceder_sticker_se_houver(db, aluno_id: str, sticker_id):
    if not sticker_id:
        return
    ja_tem = (
        db.table("aluno_stickers")
        .select("sticker_id")
        .eq("aluno_id", aluno_id)
        .eq("sticker_id", sticker_id)
        .execute()
    )
    if not ja_tem.data:
        db.table("aluno_stickers").insert({"aluno_id": aluno_id, "sticker_id": sticker_id}).execute()


@app.route("/api/professor/entregas/<progresso_id>/corrigir", methods=["POST"])
@token_required
@professor_required
def corrigir_entrega(current_user, progresso_id):
    """
    Corrige uma entrega. Se a entrega pertence a uma missão de equipe
    (progresso.equipe_id preenchido), a MESMA nota e feedback são aplicados
    a TODOS os membros da equipe (cada um com sua própria linha em
    progresso_missoes, todas marcadas nota_origem='equipe') — é o
    lançamento "em grupo" pedido: todos recebem a mesma nota de uma vez,
    mas o professor pode depois ajustar a nota de um aluno específico via
    PATCH /api/professor/entregas/<id>/nota-individual, sem afetar os
    demais.
    """
    db = get_user_supabase()
    progresso = _progresso_da_sala_professor(db, progresso_id, current_user["id"])
    if not progresso:
        return _error("Entrega não encontrada ou sem permissão.", 404)

    body = request.get_json() or {}
    try:
        nota = float(body.get("nota", ""))
        if not (0 <= nota <= 10):
            raise ValueError()
    except (ValueError, TypeError):
        return _error("Nota inválida. Informe um número entre 0 e 10.")

    feedback = body.get("feedback_professor", "").strip()
    sticker_id = progresso["missoes"].get("sticker_recompensa_id")
    equipe_id = progresso.get("equipe_id")

    try:
        dados_correcao = {
            "nota": nota,
            "comentario_professor": feedback,
            "status": "corrigido",
            "validada_professor": True,
            "nota_origem": "equipe" if equipe_id else "individual",
        }

        if equipe_id:
            # Lançamento em grupo: todas as entregas desta equipe para esta
            # mesma missão recebem a mesma nota/feedback de uma vez.
            upd = (
                db.table("progresso_missoes")
                .update(dados_correcao)
                .eq("missao_id", progresso["missao_id"])
                .eq("equipe_id", equipe_id)
                .execute()
            )
            if not upd.data:
                return _error("Não foi possível corrigir esta entrega.", 403)
            for linha in upd.data:
                _conceder_sticker_se_houver(db, linha["aluno_id"], sticker_id)
            return _success({
                "message": f"Entrega corrigida! Nota lançada para os {len(upd.data)} membros da equipe.",
                "membros_afetados": len(upd.data),
            })

        # Missão individual: corrige só esta linha.
        upd = (
            db.table("progresso_missoes")
            .update(dados_correcao)
            .eq("id", progresso_id)
            .execute()
        )
        # 🔒 Camada 2, reforço pós-escrita: se 0 linhas foram afetadas (ex.:
        # posse mudou entre a checagem e o update), não reportamos sucesso.
        if not upd.data:
            return _error("Não foi possível corrigir esta entrega.", 403)

        _conceder_sticker_se_houver(db, progresso["aluno_id"], sticker_id)
        return _success({"message": "Entrega corrigida e nota lançada!"})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/entregas/<progresso_id>/nota-individual", methods=["PATCH"])
@token_required
@professor_required
def corrigir_nota_individual(current_user, progresso_id):
    """
    Sobrescreve a nota/feedback de UM aluno específico, sem afetar os
    demais membros da equipe (mesmo quando a entrega original foi lançada
    em grupo por corrigir_entrega). Marca nota_origem='individual' nesta
    linha, para o front poder sinalizar "nota ajustada manualmente" na UI.

    Só funciona em entregas já corrigidas (validada_professor=True) — para
    a primeira correção, use POST .../corrigir (em grupo ou individual,
    conforme a modalidade da missão).
    """
    db = get_user_supabase()
    progresso = _progresso_da_sala_professor(db, progresso_id, current_user["id"])
    if not progresso:
        return _error("Entrega não encontrada ou sem permissão.", 404)

    if not progresso.get("validada_professor"):
        return _error("Esta entrega ainda não foi corrigida. Use a correção normal primeiro.", 409)

    body = request.get_json() or {}
    try:
        nota = float(body.get("nota", ""))
        if not (0 <= nota <= 10):
            raise ValueError()
    except (ValueError, TypeError):
        return _error("Nota inválida. Informe um número entre 0 e 10.")

    feedback = body.get("feedback_professor")

    dados = {"nota": nota, "nota_origem": "individual"}
    if feedback is not None:
        dados["comentario_professor"] = feedback.strip()

    try:
        upd = db.table("progresso_missoes").update(dados).eq("id", progresso_id).execute()
        if not upd.data:
            return _error("Não foi possível ajustar esta nota.", 403)
        return _success({"message": "Nota individual ajustada.", "progresso": upd.data[0]})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — PAUTA / DESEMPENHO
# =============================================================================
@app.route("/api/professor/salas/<sala_id>/pauta", methods=["GET"])
@token_required
@professor_required
def pauta(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    periodo_id = request.args.get("periodo_id")

    periodos = db.table("periodos").select("id, nome, meta_missoes").eq("sala_id", sala_id).order("criado_em").execute().data

    # Pauta só considera missões PUBLICADAS — um rascunho não tem nota
    # ainda (o aluno nem vê o conteúdo dele), então não faz sentido entrar
    # no cálculo da média.
    q_missoes = db.table("missoes").select("id, titulo, peso_nota, ordem").eq("sala_id", sala_id).eq("status", "publicada")
    if periodo_id:
        q_missoes = q_missoes.eq("periodo_id", periodo_id)
    missoes = q_missoes.order("ordem").execute().data

    q_quiz = db.table("quiz").select("id, titulo, periodo_id").eq("sala_id", sala_id)
    if periodo_id:
        q_quiz = q_quiz.eq("periodo_id", periodo_id)
    quizzes = q_quiz.order("criado_em").execute().data

    alunos_r = db.table("aluno_salas").select("perfis(id, nome)").eq("sala_id", sala_id).execute().data
    alunos = [item["perfis"] for item in alunos_r if item.get("perfis")]

    missao_ids = [m["id"] for m in missoes]
    progressos = (
        db.table("progresso_missoes").select("*").in_("missao_id", missao_ids).execute().data
        if missao_ids else []
    )

    quiz_ids = [q["id"] for q in quizzes]
    quiz_progressos = (
        db.table("quiz_progress")
        .select("aluno_id, quiz_id, score, correct_answers, total_questions, is_late")
        .in_("quiz_id", quiz_ids)
        .execute().data
        if quiz_ids else []
    )
    qprog_idx = {(qp["aluno_id"], qp["quiz_id"]): qp for qp in quiz_progressos}
    aluno_equipe_map = _build_equipe_map_periodo(db, sala_id, periodo_id)

    pauta_data = []
    for aluno in alunos:
        aid = aluno["id"]
        notas = {}
        soma_m, peso_m = 0.0, 0.0
        for m in missoes:
            prog = next((p for p in progressos if p["aluno_id"] == aid and p["missao_id"] == m["id"]), None)
            nota = prog["nota"] if prog and prog.get("nota") is not None else None
            notas[m["id"]] = {
                "nota": nota,
                "nota_origem": prog.get("nota_origem") if prog else None,
            }
            if nota is not None:
                peso = float(m.get("peso_nota") or 1)
                soma_m += nota * peso
                peso_m += peso
        media_missoes = round(soma_m / peso_m, 2) if peso_m > 0 else None

        quiz_notas = {}
        soma_q, count_q = 0.0, 0
        for q in quizzes:
            qp = qprog_idx.get((aid, q["id"]))
            nota_q = round(qp["score"] / 10, 1) if qp else None
            quiz_notas[q["id"]] = nota_q
            if nota_q is not None:
                soma_q += nota_q
                count_q += 1
        media_quizzes = round(soma_q / count_q, 2) if count_q > 0 else None

        partes = [x for x in [media_missoes, media_quizzes] if x is not None]
        media_final = round(sum(partes) / len(partes), 2) if partes else None

        pauta_data.append({
            "aluno": aluno,
            "notas": notas,
            "quiz_notas": quiz_notas,
            "media_missoes": media_missoes,
            "media_quizzes": media_quizzes,
            "media": media_final,
            "equipe_nome": aluno_equipe_map.get(aid, "—"),
        })

    return _success({
        "pauta_data": pauta_data,
        "missoes": missoes,
        "quizzes": quizzes,
        "periodos": periodos,
    })


@app.route("/api/professor/salas/<sala_id>/desempenho-consolidado", methods=["GET"])
@token_required
@professor_required
def desempenho_consolidado(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    periodo_id = request.args.get("periodo_id")
    periodos = db.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data

    try:
        # ⚠️ NOTA DE SEGURANÇA (schema, fora do escopo deste arquivo):
        # `vw_desempenho_consolidado` é uma VIEW comum. Se ela não foi criada
        # com `security_invoker = true` (Postgres 15+), views no Supabase
        # tendem a herdar os privilégios de quem as criou (normalmente o
        # role `postgres`, que tem BYPASSRLS) — ou seja, a query pode
        # devolver linhas de OUTRAS salas mesmo com RLS habilitado nas
        # tabelas de origem. O filtro explícito por `sala_id` abaixo (já
        # validado por `_sala_do_professor`) é a proteção REAL nesta rota.
        # Recomendo rodar, uma vez, no banco:
        #   ALTER VIEW public.vw_desempenho_consolidado SET (security_invoker = true);
        q = db.table("vw_desempenho_consolidado").select("*").eq("sala_id", sala_id)
        if periodo_id:
            missoes_periodo = (
                db.table("missoes").select("id").eq("sala_id", sala_id).eq("periodo_id", periodo_id).execute().data
            )
            ids_missoes = [m["id"] for m in missoes_periodo]
            if not ids_missoes:
                return _success({"registros": [], "periodos": periodos})
            q = q.in_("missao_id", ids_missoes)
        registros = q.order("aluno_nome").execute().data
        return _success({"registros": registros, "periodos": periodos})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — BIBLIOTECA DE MATERIAIS
# =============================================================================
_BIBLIO_EXTENSOES_PERMITIDAS = {"pdf", "png", "jpg", "jpeg", "webp"}


def _ler_corpo_material():
    """
    Mesma lógica de _ler_corpo_missao: aceita JSON (sem anexo) ou
    multipart/form-data (com anexo opcional no campo "anexo"). `links`, em
    multipart, chega como string JSON (já que form-data não tem tipos
    aninhados) e precisa ser decodificado manualmente.
    """
    if request.content_type and "multipart/form-data" in request.content_type:
        body = request.form
        anexo = request.files.get("anexo")
        links_raw = body.get("links")
        try:
            links = json.loads(links_raw) if links_raw else []
        except (TypeError, ValueError):
            links = []
        return body, anexo, links
    body = request.get_json(silent=True) or {}
    return body, None, (body.get("links") or [])


def _upload_anexo_biblioteca(db, arquivo):
    """
    Upload de PDF/imagem para a biblioteca, no bucket compartilhado
    "anexos_geral", prefixo "biblioteca/". Reaproveita a validação de
    tamanho (20 MB) de _upload_anexo_geral, mas restringe as extensões
    aceitas a PDF/imagem (um material de biblioteca não é um .zip ou .py).
    """
    if not arquivo or not arquivo.filename:
        return None, None, "Nenhum arquivo selecionado."
    extensao = arquivo.filename.rsplit(".", 1)[-1].lower() if "." in arquivo.filename else ""
    if extensao not in _BIBLIO_EXTENSOES_PERMITIDAS:
        permitidas = ", ".join(sorted(_BIBLIO_EXTENSOES_PERMITIDAS))
        return None, None, f"Tipo de arquivo não permitido para a biblioteca. Use um dos formatos: {permitidas}."
    url, erro = _upload_anexo_geral(db, arquivo, "biblioteca/")
    if erro:
        return None, None, erro
    return url, arquivo.filename, None


@app.route("/api/professor/salas/<sala_id>/biblioteca", methods=["GET"])
@token_required
@professor_required
def biblioteca_professor(current_user, sala_id):
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    missoes = db.table("missoes").select("id, titulo, ordem").eq("sala_id", sala_id).order("ordem").execute().data

    q = (
        db.table("biblioteca_materiais")
        .select("*, missoes(titulo)")
        .eq("sala_id", sala_id)
        .order("criado_em", desc=True)
    )
    missao_filtro = request.args.get("missao_id", "")
    if missao_filtro:
        q = q.eq("missao_id", missao_filtro)
    status_filtro = request.args.get("status", "")
    if status_filtro:
        q = q.eq("status", status_filtro)
    materiais = q.execute().data

    return _success({
        "missoes": missoes,
        "materiais": materiais,
        "pendentes_aprovacao": sum(1 for m in materiais if m.get("status") == "pendente"),
    })


@app.route("/api/professor/salas/<sala_id>/biblioteca", methods=["POST"])
@token_required
@professor_required
def criar_material(current_user, sala_id):
    """Professor cria material já aprovado. Aceita anexo (PDF/imagem) opcional."""
    db = get_user_supabase()
    if not _sala_do_professor(db, sala_id, current_user["id"]):
        return _error("Sala não encontrada ou sem permissão.", 404)

    body, anexo, links = _ler_corpo_material()
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título do material é obrigatório.")

    missao_id = body.get("missao_id") or None
    if missao_id and not _missao_da_sala(db, missao_id, sala_id, current_user["id"]):
        return _error("Missão inválida para esta sala.", 404)

    anexo_url, anexo_nome = None, None
    if anexo and anexo.filename:
        anexo_url, anexo_nome, erro_anexo = _upload_anexo_biblioteca(db, anexo)
        if erro_anexo:
            return _error(erro_anexo, 400)

    try:
        r = db.table("biblioteca_materiais").insert({
            "professor_id": current_user["id"],
            "sala_id": sala_id,
            "missao_id": missao_id,
            "titulo": titulo,
            "descricao": body.get("descricao") or None,
            "links": links,
            "anexo_url": anexo_url,
            "anexo_nome": anexo_nome,
            "status": "aprovado",
            "criado_por": current_user["id"],
            "criado_por_role": "professor",
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/biblioteca/<material_id>", methods=["PUT"])
@token_required
@professor_required
def editar_material(current_user, sala_id, material_id):
    """
    Professor edita o material — inclusive uma proposta de aluno pendente,
    o que já serve como "aprovar com ajustes": basta mandar status='aprovado'
    no corpo (ou deixar status de fora para só trocar o conteúdo sem mudar
    o status atual).
    """
    db = get_user_supabase()
    material_atual = _material_da_sala(db, material_id, sala_id, current_user["id"])
    if not material_atual:
        return _error("Material não encontrado ou sem permissão.", 404)

    body, anexo, links = _ler_corpo_material()
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título do material é obrigatório.")

    missao_id = body.get("missao_id") or None
    if missao_id and not _missao_da_sala(db, missao_id, sala_id, current_user["id"]):
        return _error("Missão inválida para esta sala.", 404)

    status_novo = (body.get("status") or material_atual.get("status") or "aprovado").strip().lower()
    if status_novo not in ("pendente", "aprovado"):
        return _error("Status inválido. Use 'pendente' ou 'aprovado'.", 400)

    dados = {
        "titulo": titulo,
        "descricao": body.get("descricao") or None,
        "missao_id": missao_id,
        "links": links,
        "status": status_novo,
    }
    if status_novo == "aprovado" and material_atual.get("status") != "aprovado":
        dados["aprovado_por"] = current_user["id"]
        dados["aprovado_em"] = datetime.now(timezone.utc).isoformat()

    if anexo and anexo.filename:
        anexo_url, anexo_nome, erro_anexo = _upload_anexo_biblioteca(db, anexo)
        if erro_anexo:
            return _error(erro_anexo, 400)
        dados["anexo_url"] = anexo_url
        dados["anexo_nome"] = anexo_nome

    try:
        r = db.table("biblioteca_materiais").update(dados).eq("id", material_id).execute()
        return _success(r.data[0] if r.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/biblioteca/<material_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_material(current_user, sala_id, material_id):
    db = get_user_supabase()
    if not _material_da_sala(db, material_id, sala_id, current_user["id"]):
        return _error("Material não encontrado ou sem permissão.", 404)
    try:
        db.table("biblioteca_materiais").delete().eq("id", material_id).execute()
        return _success({"message": "Material removido."})
    except Exception as e:
        return _error(str(e), 500)


# =============================================================================
# PROFESSOR — GERAÇÃO DE QUIZ COM GEMINI (não acessa dados de sala/aluno)
# =============================================================================
@app.route("/api/professor/gemini/gerar-quiz", methods=["POST"])
@token_required
@professor_required
def gerar_quiz_gemini(current_user):
    if genai_client is None:
        return _error("Serviço de geração de quiz indisponível (GEMINI_API_KEY não configurada).", 500)

    body = request.get_json() or {}
    tema = body.get("tema", "").strip()
    quantidade = int(body.get("quantidade", 5) or 5)
    dificuldade = body.get("dificuldade", "medio")

    if not tema:
        return _error("Informe o tema do quiz.")

    prompt = (
        f"Gere {quantidade} perguntas de múltipla escolha sobre '{tema}', "
        f"nível de dificuldade '{dificuldade}'. Responda SOMENTE em JSON, "
        "no formato: "
        '{"perguntas": [{"pergunta": "...", "opcoes": ["A", "B", "C", "D"], "resposta_correta": "A"}]}'
    )

    try:
        response = genai_client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        dados = json.loads(response.text)
        return _success(dados)
    except Exception as e:
        return _error(f"Erro ao gerar quiz: {str(e)}", 500)


# =============================================================================
# ALUNO — DASHBOARD / TRILHA
# =============================================================================
@app.route("/api/aluno/dashboard", methods=["GET"])
@token_required
def dashboard_aluno(current_user):
    db = get_user_supabase()
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(db, aluno_id)

    if not sala_id:
        return _success({"tem_sala": False})

    sala_r = (
        db.table("salas")
        .select("id, nome, codigo_acesso, perfis!salas_professor_id_fkey(nome)")
        .eq("id", sala_id)
        .execute()
    )
    sala = sala_r.data[0] if sala_r.data else None

    periodos_sala = db.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data
    equipes_sala = db.table("equipes").select("id, nome, periodo_id").eq("sala_id", sala_id).execute().data
    equipe_ids_sala = [e["id"] for e in equipes_sala]
    equipes_map = {e["id"]: e for e in equipes_sala}

    equipe = None
    equipe_periodo_nome = None
    colegas = []

    if equipe_ids_sala:
        membro_r = (
            db.table("equipe_membros")
            .select("equipe_id")
            .eq("aluno_id", aluno_id)
            .in_("equipe_id", equipe_ids_sala)
            .execute().data
        )
        if membro_r:
            periodo_order = {p["id"]: i for i, p in enumerate(periodos_sala)}
            equipes_aluno = [equipes_map[m["equipe_id"]] for m in membro_r if m["equipe_id"] in equipes_map]
            equipes_aluno.sort(key=lambda e: periodo_order.get(e.get("periodo_id"), -1), reverse=True)
            equipe_atual = equipes_aluno[0] if equipes_aluno else None

            if equipe_atual:
                equipe = equipe_atual
                periodo_ativo_id = equipe_atual.get("periodo_id")
                periodo_nomes = {p["id"]: p["nome"] for p in periodos_sala}
                equipe_periodo_nome = periodo_nomes.get(periodo_ativo_id, "")

                membros_r = db.table("equipe_membros").select("perfis(id, nome)").eq("equipe_id", equipe["id"]).execute().data
                colegas = [
                    {"id": m["perfis"]["id"], "nome": m["perfis"]["nome"]}
                    for m in membros_r
                    if m.get("perfis") and m["perfis"]["id"] != aluno_id
                ]

    periodos_trilha, total, concluidas = _build_periodos_trilha(db, aluno_id, sala_id)

    all_quizzes = (
        db.table("quiz")
        .select("id, titulo, dificuldade, xp_reward")
        .eq("sala_id", sala_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    quiz_ids_all = [q["id"] for q in all_quizzes]
    respondidos_ids = set()
    if quiz_ids_all:
        resp_r = (
            db.table("quiz_progress")
            .select("quiz_id")
            .eq("aluno_id", aluno_id)
            .in_("quiz_id", quiz_ids_all)
            .execute().data
        )
        respondidos_ids = {r["quiz_id"] for r in resp_r}
    quizzes_disponiveis = [q for q in all_quizzes if q["id"] not in respondidos_ids]

    return _success({
        "tem_sala": True,
        "sala": sala,
        "equipe": equipe,
        "equipe_periodo_nome": equipe_periodo_nome,
        "colegas": colegas,
        "periodos_trilha": periodos_trilha,
        "total": total,
        "concluidas": concluidas,
        "quizzes_disponiveis": quizzes_disponiveis,
    })

def _biblioteca_da_sala(db, sala_id, missao_id_filtro=None, aluno_id=None):
    """
    Materiais da biblioteca de uma sala, do ponto de vista do ALUNO: só
    materiais 'aprovado' (visíveis a toda a turma) + as PRÓPRIAS propostas
    'pendente' do aluno (para ele acompanhar o status da sua sugestão,
    mesmo antes de aprovada). O RLS (policy "biblio_select") já impõe essa
    mesma regra como camada 2, então mesmo que este filtro tivesse um erro
    aqui, o banco não devolveria material pendente de outro aluno.
    """
    q = (
        db.table("biblioteca_materiais")
        .select("*, missoes(id, titulo, ordem)")
        .eq("sala_id", sala_id)
        .order("criado_em", desc=True)
    )
    if missao_id_filtro:
        q = q.eq("missao_id", missao_id_filtro)
    materiais = q.execute().data
    return [
        m for m in materiais
        if m.get("status") == "aprovado" or (aluno_id and m.get("criado_por") == aluno_id)
    ]


@app.route("/api/aluno/biblioteca", methods=["GET"])
@token_required
def biblioteca_aluno(current_user):
    db = get_user_supabase()
    sala_id = _get_vinculo_aluno(db, current_user["id"])

    if not sala_id:
        return _success({"materiais": []})

    materiais = _biblioteca_da_sala(db, sala_id, request.args.get("missao_id", ""), current_user["id"])
    return _success({"materiais": materiais})


@app.route("/api/aluno/biblioteca", methods=["POST"])
@token_required
def propor_material_aluno(current_user):
    """
    Aluno propõe um material para a biblioteca: entra como status='pendente'
    e só fica visível para a turma depois que o professor aprovar (PUT
    .../biblioteca/<id> com status='aprovado'). O próprio aluno continua
    vendo a própria proposta enquanto pendente (ver _biblioteca_da_sala).

    Aceita multipart/form-data (com anexo PDF/imagem) ou JSON (só links/texto).
    """
    db = get_user_supabase()
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(db, aluno_id)
    if not sala_id:
        return _error("Você não está matriculado em nenhuma sala.", 403)

    body, anexo, links = _ler_corpo_material()
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título do material é obrigatório.")

    missao_id = body.get("missao_id") or None
    if missao_id:
        missao_r = db.table("missoes").select("id").eq("id", missao_id).eq("sala_id", sala_id).execute()
        if not missao_r.data:
            return _error("Missão inválida para esta sala.", 404)

    anexo_url, anexo_nome = None, None
    if anexo and anexo.filename:
        anexo_url, anexo_nome, erro_anexo = _upload_anexo_biblioteca(db, anexo)
        if erro_anexo:
            return _error(erro_anexo, 400)

    # professor_id é obrigatório na tabela (legado) — preenchido aqui com o
    # professor da sala, já que quem está propondo é o aluno (criado_por).
    sala_r = db.table("salas").select("professor_id").eq("id", sala_id).execute()
    if not sala_r.data:
        return _error("Sala não encontrada.", 404)
    professor_id = sala_r.data[0]["professor_id"]

    try:
        r = db.table("biblioteca_materiais").insert({
            "professor_id": professor_id,
            "sala_id": sala_id,
            "missao_id": missao_id,
            "titulo": titulo,
            "descricao": body.get("descricao") or None,
            "links": links,
            "anexo_url": anexo_url,
            "anexo_nome": anexo_nome,
            "status": "pendente",
            "criado_por": aluno_id,
            "criado_por_role": "aluno",
        }).execute()
        return _success({
            "material": r.data[0],
            "message": "Material enviado para aprovação do professor.",
        }, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/salas/<sala_id>/biblioteca", methods=["GET"])
@token_required
def biblioteca_aluno_por_sala(current_user, sala_id):
    """
    Alias de `/api/aluno/biblioteca` que aceita `sala_id` na URL — existe só
    por compatibilidade com o app mobile, que chama esse padrão (o mesmo da
    rota do professor). O `sala_id` do path é tratado como não-confiável:
    a fonte de verdade continua sendo o vínculo real do aluno
    (`_get_vinculo_aluno`); se não bater, 403. Isso evita que o aluno consiga
    "espiar" a biblioteca de outra sala só editando a URL — mesmo que o RLS
    já bloqueasse a leitura, é a mesma camada 2 de checagem explícita usada
    no restante do arquivo (ver `_sala_do_professor` e afins).

    Prefira migrar o app para `/api/aluno/biblioteca` (sem sala_id) quando
    possível: o aluno só pertence a 1 sala, então o path param é redundante
    e essa rota pode ser removida no futuro.
    """
    db = get_user_supabase()
    sala_vinculada = _get_vinculo_aluno(db, current_user["id"])
    if not sala_vinculada or sala_vinculada != sala_id:
        return _error("Sala não encontrada ou sem permissão.", 403)

    materiais = _biblioteca_da_sala(db, sala_id, request.args.get("missao_id", ""), current_user["id"])
    return _success({"materiais": materiais})

@app.route("/api/aluno/missoes/<missao_id>", methods=["GET"])
@token_required
def ver_missao_aluno(current_user, missao_id):
    """
    Consulta de uma missão pelo aluno. Usa `_missao_desbloqueada` — a MESMA
    regra de desbloqueio usada na trilha (_build_periodos_trilha) e no envio
    (enviar_missao) — para que as três rotas nunca mais divirjam sobre se
    esta missão está ou não liberada para o aluno.

    Se a missão está em rascunho, devolve o formato "bloqueado" (sem
    conteúdo) do mesmo jeito que a trilha — mesma regra, mesmo resultado,
    não importa por qual rota o aluno chegou até ela.
    """
    db = get_user_supabase()
    aluno_id = current_user["id"]

    missao_r = (
        db.table("missoes")
        .select("*, stickers(imagem_url, nome, raridade), salas(id, nome)")
        .eq("id", missao_id)
        .execute()
    )
    missao = missao_r.data[0] if missao_r.data else None
    if not missao:
        return _error("Missão não encontrada.", 404)

    sala_id = missao["salas"]["id"]
    # 🔒 Confirma matrícula do aluno NESTA sala antes de liberar a missão
    if _get_vinculo_aluno(db, aluno_id) != sala_id:
        return _error("Acesso negado.", 403)

    if missao.get("status") == "rascunho":
        missao_out = _serializar_missao_para_aluno(missao, desbloqueada=False)
        return _success({
            "missao": missao_out,
            "progresso": None,
            "bloqueada": True,
            "sala_id": sala_id,
            "materiais": [],
            "equipe": None,
        })

    desbloqueada = _missao_desbloqueada(db, aluno_id, missao_id)

    prog_r = (
        db.table("progresso_missoes")
        .select("*")
        .eq("aluno_id", aluno_id)
        .eq("missao_id", missao_id)
        .execute()
    )
    progresso = prog_r.data[0] if prog_r.data else None

    # Modalidade 'equipe': o aluno precisa ter equipe no período da missão
    # para visualizar o conteúdo completo e enviar (decisão de produto).
    equipe = None
    if missao.get("modalidade") == "equipe":
        equipe = _equipe_do_aluno_no_periodo(db, aluno_id, sala_id, missao.get("periodo_id"))

    materiais = (
        db.table("biblioteca_materiais")
        .select("id, titulo, descricao, links, anexo_url, anexo_nome")
        .eq("missao_id", missao_id)
        .eq("status", "aprovado")
        .order("criado_em")
        .execute().data
    )

    anexos_entrega = []
    if progresso:
        anexos_entrega = (
            db.table("progresso_missoes_anexos")
            .select("id, url, nome_arquivo, criado_em")
            .eq("progresso_id", progresso["id"])
            .order("criado_em")
            .execute().data
        )

    return _success({
        "missao": {**missao, "desbloqueada": desbloqueada},
        "progresso": progresso,
        "anexos_entrega": anexos_entrega,
        "bloqueada": not desbloqueada,
        "sala_id": sala_id,
        "materiais": materiais,
        "equipe": equipe,
    })


_MAX_ANEXOS_ENTREGA = 3


@app.route("/api/aluno/missoes/enviar", methods=["POST"])
@token_required
def enviar_missao(current_user):
    """
    Envia (ou reenvia) a entrega de uma missão: texto + até 3 anexos.

    Aceita multipart/form-data: `missao_id`, `texto` (opcional) e até 3
    arquivos em `request.files.getlist("anexos")`. Também aceita JSON puro
    (sem anexo, só texto), para compatibilidade com entregas que não
    exigem arquivo.

    Unificação do desbloqueio: usa `_missao_desbloqueada` — a MESMA regra
    de ver_missao_aluno e _build_periodos_trilha — então uma missão que
    aparece bloqueada na trilha ou na consulta NUNCA pode ser enviada por
    aqui (antes desta correção, enviar_missao não checava desbloqueio
    nenhum).

    Modalidade 'equipe': a entrega é gravada para TODOS os membros da
    equipe do aluno nesse período (uma linha de progresso por membro, cada
    uma com o mesmo equipe_id/texto/anexos) — assim a trigger de XP e o RLS
    por aluno continuam funcionando sem caso especial, e cada membro vê a
    entrega no próprio progresso.
    """
    db = get_user_supabase()
    aluno_id = current_user["id"]

    if request.content_type and "multipart/form-data" in request.content_type:
        body = request.form
        anexos = [f for f in request.files.getlist("anexos") if f and f.filename][:_MAX_ANEXOS_ENTREGA + 1]
    else:
        body = request.get_json(silent=True) or {}
        anexos = []

    missao_id = body.get("missao_id")
    if not missao_id:
        return _error("Informe a missão.")

    texto_entrega = (body.get("texto") or "").strip() or None

    if len(anexos) > _MAX_ANEXOS_ENTREGA:
        return _error(f"No máximo {_MAX_ANEXOS_ENTREGA} arquivos por entrega.", 400)

    if not texto_entrega and not anexos:
        return _error("Envie um texto ou pelo menos um arquivo para a entrega.", 400)

    try:
        missao_r = (
            db.table("missoes")
            .select("id, sala_id, periodo_id, data_limite, status, modalidade")
            .eq("id", missao_id)
            .execute()
        )
        if not missao_r.data:
            return _error("Missão não encontrada.", 404)
        missao = missao_r.data[0]

        # 🔒 Confirma matrícula do aluno na sala da missão
        if _get_vinculo_aluno(db, aluno_id) != missao["sala_id"]:
            return _error("Acesso negado.", 403)

        if missao.get("status") != "publicada":
            return _error("Esta missão ainda não está disponível.", 403)

        # 🔒 REGRA ÚNICA de desbloqueio — mesma usada na trilha e na
        # consulta da missão. Sem isso, dava para enviar uma missão "mais à
        # frente" sem ter completado as anteriores.
        if not _missao_desbloqueada(db, aluno_id, missao_id):
            return _error("Complete as missões anteriores (e aguarde a correção do professor) antes de enviar esta.", 403)

        # Modalidade 'equipe': aluno precisa ter equipe no período da missão.
        alunos_destino = [aluno_id]
        equipe_id = None
        if missao.get("modalidade") == "equipe":
            equipe = _equipe_do_aluno_no_periodo(db, aluno_id, missao["sala_id"], missao.get("periodo_id"))
            if not equipe:
                return _error("Você precisa estar em uma equipe para enviar esta missão.", 403)
            equipe_id = equipe["id"]
            alunos_destino = [m["id"] for m in equipe["membros"]] or [aluno_id]

        # Se QUALQUER membro já teve a entrega corrigida, ninguém mais pode reenviar.
        existentes = (
            db.table("progresso_missoes")
            .select("id, aluno_id, status")
            .eq("missao_id", missao_id)
            .in_("aluno_id", alunos_destino)
            .execute()
        ).data or []
        if any(e["status"] == "corrigido" for e in existentes):
            return _error("Esta missão já foi corrigida e não pode ser reenviada.", 409)
        existentes_por_aluno = {e["aluno_id"]: e for e in existentes}

        # 🔒 Anexos (até 3) vão para o bucket compartilhado "anexos_geral",
        # sob o prefixo "entregas/", cada um com nome gerado via uuid4().
        anexos_enviados = []  # [{"url":..., "nome_arquivo":...}, ...]
        for arquivo in anexos:
            url, erro_anexo = _upload_anexo_geral(db, arquivo, "entregas/")
            if erro_anexo:
                return _error(erro_anexo, 400)
            anexos_enviados.append({"url": url, "nome_arquivo": arquivo.filename})

        atrasado = False
        dl = _parse_dt(missao.get("data_limite"))
        if dl and datetime.now(timezone.utc) > dl:
            atrasado = True

        agora = datetime.now(timezone.utc).isoformat()
        dados_entrega = {
            "status": "entregue",
            "entregue_em": agora,
            "texto_entrega": texto_entrega,
            "equipe_id": equipe_id,
        }
        if anexos_enviados:
            # url_entrega (coluna legada) espelha o 1º anexo, para quem
            # ainda lê só esse campo; a lista completa fica em
            # progresso_missoes_anexos.
            dados_entrega["url_entrega"] = anexos_enviados[0]["url"]

        # 🔒 Um INSERT/UPDATE por aluno destino (sempre a partir de
        # alunos_destino, nunca de um campo livre do body) — em missão
        # individual, alunos_destino == [aluno_id][o próprio current_user].
        progresso_ids_por_aluno = {}
        for destino_id in alunos_destino:
            existente = existentes_por_aluno.get(destino_id)
            if existente:
                db.table("progresso_missoes").update(dados_entrega).eq("id", existente["id"]).execute()
                progresso_ids_por_aluno[destino_id] = existente["id"]
            else:
                r = db.table("progresso_missoes").insert({
                    "aluno_id": destino_id,
                    "missao_id": missao_id,
                    **dados_entrega,
                }).execute()
                progresso_ids_por_aluno[destino_id] = r.data[0]["id"]

        # Anexos: substitui os anteriores (delete + insert) em CADA
        # progresso afetado, para refletir a mesma lista de arquivos em
        # todos os membros da equipe.
        if anexos_enviados:
            for progresso_id in progresso_ids_por_aluno.values():
                db.table("progresso_missoes_anexos").delete().eq("progresso_id", progresso_id).execute()
                db.table("progresso_missoes_anexos").insert([
                    {"progresso_id": progresso_id, "url": a["url"], "nome_arquivo": a["nome_arquivo"]}
                    for a in anexos_enviados
                ]).execute()

        return _success({
            "message": "Missão enviada! Aguarde a correção do professor. 🚀",
            "atrasado": atrasado,
            "enviado_para_equipe": equipe_id is not None,
        })
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/entrar-sala", methods=["POST"])
@token_required
def entrar_sala_por_codigo(current_user):
    db = get_user_supabase()
    aluno_id = current_user["id"]
    body = request.get_json() or {}
    codigo = body.get("codigo_sala", "").strip().upper()
    if not codigo:
        return _error("Informe o código da sala.")

    try:
        sala_r = db.table("salas").select("id, nome").eq("codigo_acesso", codigo).execute()
        if not sala_r.data:
            return _error("Código inválido. Verifique com seu professor.", 404)
        sala = sala_r.data[0]

        ja_tem = db.table("aluno_salas").select("sala_id").eq("aluno_id", aluno_id).execute()
        if ja_tem.data:
            return _error("Você já está matriculado em uma sala.", 409)

        # 🔒 aluno_id vem sempre de current_user
        db.table("aluno_salas").insert({"aluno_id": aluno_id, "sala_id": sala["id"]}).execute()
        return _success({"message": f"Você entrou na sala '{sala['nome']}'!", "sala": sala}, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/aluno/quiz/<quiz_id>/responder", methods=["POST"])
@token_required
def responder_quiz(current_user, quiz_id):
    db = get_user_supabase()
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(db, aluno_id)
    if not sala_id:
        return _error("Você não está matriculado em nenhuma sala.", 403)

    quiz_r = db.table("quiz").select("*").eq("id", quiz_id).execute()
    quiz = quiz_r.data[0] if quiz_r.data else None
    # 🔒 Confirma que o quiz pertence à sala do aluno
    if not quiz or quiz["sala_id"] != sala_id:
        return _error("Quiz não encontrado ou sem acesso.", 404)

    ja_existe = (
        db.table("quiz_progress")
        .select("id")
        .eq("aluno_id", aluno_id)
        .eq("quiz_id", quiz_id)
        .execute()
    )
    if ja_existe.data:
        return _error("Você já respondeu este quiz.", 409)

    body = request.get_json() or {}
    respostas = body.get("respostas", {})

    perguntas = (
        db.table("question")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("ordem")
        .execute().data
    )
    if not perguntas:
        return _error("Este quiz não possui perguntas.")

    correct_answers = sum(
        1 for p in perguntas
        if str(respostas.get(p["id"], "")).strip().lower() == str(p.get("correct_answer", "")).strip().lower()
    )
    total_questions = len(perguntas)
    score = round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0.0

    try:
        # 🔒 aluno_id vem sempre de current_user; UNIQUE(aluno_id, quiz_id)
        # no banco + a policy `qprog_insert` (aluno_id = auth.uid()) reforçam.
        db.table("quiz_progress").insert({
            "aluno_id": aluno_id,
            "quiz_id": quiz_id,
            "score": score,
            "correct_answers": correct_answers,
            "total_questions": total_questions,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "is_late": False,
        }).execute()
        return _success({
            "message": f"Quiz concluído! Você acertou {correct_answers}/{total_questions} ({score:.1f}%).",
            "score": score,
            "correct_answers": correct_answers,
            "total_questions": total_questions,
        })
    except Exception as e:
        return _error(str(e), 500)


if __name__ == "__main__":
    app.run(debug=True)