"""
api.py — Backend REST para a Plataforma Educacional Gamificada
============================================================
API RESTful

MUDANÇA DE PARADIGMA PRINCIPAL:
  ESQUEMA: Bearer JWT no cabeçalho Authorization + jsonify() + HTTP status codes

AUTENTICAÇÃO:
  O frontend (React) autentica diretamente no Supabase GoTrue e recebe um JWT.
  Cada requisição protegida envia:  Authorization: Bearer <access_token>
  O backend valida o token chamando supabase.auth.get_user(token), que é a forma
  recomendada pelo supabase-py v2 — não é necessário verificar a chave JWT manualmente.

CHAVE SUPABASE:
  Usamos a SERVICE ROLE KEY no backend para contornar o RLS em operações
  privilegiadas (o RLS é a proteção do banco — a API Flask é a camada de regras
  de negócio). O frontend usa a ANON KEY com o JWT do usuário.
"""

import os
import json
import functools
from datetime import datetime, timezone

import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

load_dotenv()

app = Flask(__name__)
# Garante que se FRONTEND_ORIGIN não existir, use "*" temporariamente para o app não quebrar
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "*")}})

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")  # Para o frontend React
SUPABASE_SERVICE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")   # Service Role Key — ignora RLS
PROFESSOR_MASTER_KEY = os.getenv("PROFESSOR_MASTER_KEY")   # Substitui FLASK_SECRET_KEY

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️ ALERTA CRÍTICO: SUPABASE_URL ou SUPABASE_SERVICE_KEY não foram detectadas no ambiente!")

# Inicialização segura do Supabase
supabase: Client = create_client(SUPABASE_URL or "https://placeholder.supabase.co", SUPABASE_SERVICE_KEY or "placeholder")

# Inicialização da API do Gemini baseada na versão carregada (google-genai >= 1.0.0)
# A nova SDK busca automaticamente a variável GEMINI_API_KEY do sistema.
try:
    ai_client = genai.Client()
except Exception as ai_err:
    print(f"⚠️ Erro ao instanciar o cliente Gemini GenAI: {ai_err}")
    ai_client = None


supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
gemini_client = genai.Client()


# ===========================================================================
# HELPERS GERAIS
# ===========================================================================

def _parse_dt(s):
    """Converte string ISO 8601 (com ou sem 'Z') para datetime timezone-aware."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _success(data=None, status=200):
    return jsonify({"ok": True, "data": data}), status


def _error(message, status=400, details=None):
    body = {"ok": False, "error": message}
    if details:
        body["details"] = details
    return jsonify(body), status


# ===========================================================================
# MIDDLEWARES DE AUTENTICAÇÃO
# ===========================================================================

def _get_token_from_header():
    """Extrai o Bearer token do cabeçalho Authorization."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def token_required(f):
    """
    Decorator que valida o JWT enviado pelo frontend React.
    Injeta `current_user` (dict com id, role, email) na função decorada.

    ANTES: session['user_id'] populado no login Flask
    AGORA: supabase.auth.get_user(token) valida o JWT emitido pelo GoTrue
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _get_token_from_header()
        if not token:
            return _error("Token de autenticação não fornecido.", 401)
        try:
            user_resp = supabase.auth.get_user(token)
            auth_user = user_resp.user
            if not auth_user:
                return _error("Token inválido ou expirado.", 401)

            perfil = (
                supabase.table("perfis")
                .select("id, nome, role, email")
                .eq("id", auth_user.id)
                .single()
                .execute()
            )
            if not perfil.data:
                return _error("Perfil não encontrado.", 401)

            kwargs["current_user"] = perfil.data
        except Exception as e:
            return _error(f"Falha na autenticação: {str(e)}", 401)
        return f(*args, **kwargs)
    return wrapper


def professor_required(f):
    """
    Decorator de autorização — deve ser usado APÓS @token_required.
    Garante que o usuário autenticado tem role='professor'.

    ANTES: if session.get('role') != 'professor': redirect(...)
    AGORA: HTTP 403 Forbidden com mensagem JSON
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        user = kwargs.get("current_user")
        if not user or user.get("role") != "professor":
            return _error("Acesso negado. Rota exclusiva para professores.", 403)
        return f(*args, **kwargs)
    return wrapper


# ===========================================================================
# LÓGICA DE NEGÓCIO — TRILHA (portada de _build_periodos_trilha)
# ===========================================================================

def _build_periodos_trilha(aluno_id: str, sala_id: str):
    """
    Regra de Negócio G01: Constrói a trilha de missões por período.

    Uma missão só fica 'disponivel' se a missão de ordem anterior (dentro do
    mesmo período) tiver validada_professor = true no progresso do aluno.
    Um período inteiro fica 'bloqueado' se todas as missões do período
    anterior não tiverem sido validadas.

    Retorna: (periodos_com_trilha, total_missoes, total_concluidas)
    """
    periodos = (
        supabase.table("periodos")
        .select("id, nome")
        .eq("sala_id", sala_id)
        .order("criado_em")
        .execute().data
    )
    if not periodos:
        return [], 0, 0

    periodo_ids = [p["id"] for p in periodos]
    todas_missoes = (
        supabase.table("missoes")
        .select("*, stickers(imagem_url, nome, raridade)")
        .eq("sala_id", sala_id)
        .in_("periodo_id", periodo_ids)
        .execute().data
    )

    mapa = {p["id"]: [] for p in periodos}
    for m in todas_missoes:
        if m.get("periodo_id") in mapa:
            mapa[m["periodo_id"]].append(m)
    for pid in mapa:
        mapa[pid].sort(key=lambda m: m.get("ordem", 0))

    ids_todas = [m["id"] for m in todas_missoes]
    progressos = (
        supabase.table("progresso_missoes")
        .select("*")
        .eq("aluno_id", aluno_id)
        .in_("missao_id", ids_todas)
        .execute().data
        if ids_todas else []
    )
    prog_map = {p["missao_id"]: p for p in progressos}

    agora = datetime.now(timezone.utc)
    numero_global = 0
    total = 0
    concluidas = 0
    periodo_anterior_ok = True
    periodos_com_trilha = []

    for periodo in periodos:
        missoes = mapa[periodo["id"]]
        ids_periodo = [m["id"] for m in missoes]
        periodo_bloqueado = not periodo_anterior_ok
        anterior_validada = True
        missoes_processadas = []

        for missao in missoes:
            numero_global += 1
            total += 1
            prog = prog_map.get(missao["id"])

            # RN-G01: bloqueio sequencial
            if periodo_bloqueado or not anterior_validada:
                status_visual = "bloqueada"
            elif prog and prog["status"] == "corrigido":
                status_visual = "concluida"
                concluidas += 1
            elif prog and prog["status"] == "entregue":
                status_visual = "enviada"
            else:
                status_visual = "disponivel"

            # O unlock da próxima missão depende de validada_professor=true
            anterior_validada = bool(prog and prog.get("validada_professor"))

            atrasada = False
            if missao.get("data_limite") and status_visual in ("disponivel", "enviada", "bloqueada"):
                dl = _parse_dt(missao["data_limite"])
                if dl and dl < agora:
                    atrasada = True

            missoes_processadas.append({
                **missao,
                "status_visual": status_visual,
                "progresso": prog,
                "atrasada": atrasada,
                "numero_global": numero_global,
            })

        validadas_p = {
            mid for mid in ids_periodo
            if prog_map.get(mid) and prog_map[mid].get("validada_professor")
        }
        periodo_anterior_ok = (len(validadas_p) == len(ids_periodo)) if ids_periodo else True

        periodos_com_trilha.append({
            "id": periodo["id"],
            "nome": periodo["nome"],
            "bloqueado": periodo_bloqueado,
            "missoes": missoes_processadas,
        })

    return periodos_com_trilha, total, concluidas


def _periodo_acessivel(aluno_id: str, sala_id: str, periodo_id: str) -> bool:
    """Verifica se o aluno pode acessar um período (todos os anteriores concluídos)."""
    if not periodo_id:
        return True
    periodos = (
        supabase.table("periodos")
        .select("id")
        .eq("sala_id", sala_id)
        .order("criado_em")
        .execute().data
    )
    ids_em_ordem = [p["id"] for p in periodos]
    if periodo_id not in ids_em_ordem:
        return True
    idx = ids_em_ordem.index(periodo_id)
    if idx == 0:
        return True

    ids_anteriores = ids_em_ordem[:idx]
    missoes_ant = (
        supabase.table("missoes")
        .select("id")
        .eq("sala_id", sala_id)
        .in_("periodo_id", ids_anteriores)
        .execute().data
    )
    ids_missoes_ant = [m["id"] for m in missoes_ant]
    if not ids_missoes_ant:
        return True

    progressos = (
        supabase.table("progresso_missoes")
        .select("missao_id, validada_professor")
        .eq("aluno_id", aluno_id)
        .in_("missao_id", ids_missoes_ant)
        .execute().data
    )
    validadas = {p["missao_id"] for p in progressos if p.get("validada_professor")}
    return validadas >= set(ids_missoes_ant)


def _get_vinculo_aluno(aluno_id: str):
    r = supabase.table("aluno_salas").select("sala_id").eq("aluno_id", aluno_id).execute()
    return r.data[0]["sala_id"] if r.data else None


def _build_equipe_map_periodo(sala_id: str, periodo_id):
    """Retorna {aluno_id: equipe_nome} para o período dado. 2 queries (sem N+1)."""
    if not periodo_id:
        return {}
    equipes_r = (
        supabase.table("equipes")
        .select("id, nome")
        .eq("sala_id", sala_id)
        .eq("periodo_id", periodo_id)
        .execute().data
    )
    equipe_ids = [e["id"] for e in equipes_r]
    equipes_map = {e["id"]: e["nome"] for e in equipes_r}
    if not equipe_ids:
        return {}
    membros_r = (
        supabase.table("equipe_membros")
        .select("equipe_id, aluno_id")
        .in_("equipe_id", equipe_ids)
        .execute().data
    )
    return {m["aluno_id"]: equipes_map.get(m["equipe_id"], "—") for m in membros_r}


# ===========================================================================
# ROTAS DE AUTENTICAÇÃO
# ===========================================================================

@app.route("/api/auth/cadastro", methods=["POST"])
def cadastro():
    """
    Cadastra novo usuário.
    RN-G02: Cadastro de professor exige chave mestra.
    IMPORTANTE: Usa upsert() em perfis para não colidir com o trigger
    on_auth_user_created que já cria o perfil automaticamente.
    """
    body = request.get_json() or {}
    email = body.get("email", "").strip()
    password = body.get("senha", "")
    nome = body.get("nome", "").strip()
    role = body.get("role", "aluno")
    chave_mestra = body.get("chave_mestra", "")

    if role == "professor":
        if chave_mestra != PROFESSOR_MASTER_KEY:
            return _error("Chave mestra de professor incorreta.", 403)

    try:
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            # Passa nome/role como metadata para o trigger on_auth_user_created capturar
            "options": {"data": {"nome": nome, "role": role}},
        })
        if not auth_response.user:
            return _error("Falha ao criar usuário no Supabase Auth.", 500)

        user_data = {
            "id": auth_response.user.id,
            "nome": nome,
            "role": role,
            "email": email,
        }
        # UPSERT — evita erro de chave duplicada caso o trigger já tenha criado o perfil
        supabase.table("perfis").upsert(user_data, on_conflict="id").execute()

        return _success({"message": "Cadastro realizado! Verifique seu e-mail."}, 201)
    except Exception as e:
        return _error(f"Erro ao cadastrar: {str(e)}", 400)


@app.route("/api/auth/login", methods=["POST"])
def login():
    """
    Autentica o usuário e retorna o JWT + dados do perfil.

    ANTES: session['user_id'] = response.user.id  (cookie de sessão server-side)
    AGORA: retorna { access_token, refresh_token, user: {...} }
           O React armazena o token no AuthContext (memória) e,
           opcionalmente, em localStorage para persistência entre abas.
    """
    body = request.get_json() or {}
    email = body.get("email", "")
    password = body.get("senha", "")

    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if not response.user:
            return _error("Credenciais inválidas.", 401)

        perfil = (
            supabase.table("perfis")
            .select("id, nome, role, email, avatar_url, xp")
            .eq("id", response.user.id)
            .single()
            .execute()
        )

        return _success({
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "user": perfil.data,
        })
    except Exception as e:
        return _error(f"Erro ao fazer login: {str(e)}", 401)


@app.route("/api/auth/logout", methods=["POST"])
@token_required
def logout(current_user):
    """Invalida a sessão no Supabase Auth."""
    try:
        supabase.auth.sign_out()
        return _success({"message": "Sessão encerrada."})
    except Exception:
        return _success({"message": "Sessão encerrada localmente."})


# ===========================================================================
# SALAS — PROFESSOR
# ===========================================================================

@app.route("/api/professor/salas", methods=["GET"])
@token_required
@professor_required
def listar_salas(current_user):
    professor_id = current_user["id"]
    salas = supabase.table("salas").select("*").eq("professor_id", professor_id).execute().data
    for sala in salas:
        sid = sala["id"]
        missoes_r = supabase.table("missoes").select("*", count="exact").eq("sala_id", sid).execute()
        alunos_r = supabase.table("aluno_salas").select("*", count="exact").eq("sala_id", sid).execute()
        sala["total_missoes"] = missoes_r.count or 0
        sala["total_alunos"] = alunos_r.count or 0
    return _success(salas)


@app.route("/api/professor/salas", methods=["POST"])
@token_required
@professor_required
def criar_sala(current_user):
    body = request.get_json() or {}
    nome_sala = body.get("nome", "").strip()
    if not nome_sala:
        return _error("O nome da sala é obrigatório.")
    try:
        codigo_sala = str(uuid.uuid4()).replace("-", "")[:6].upper()
        nova_sala = {
            "nome": nome_sala,
            "professor_id": current_user["id"],
            "codigo_acesso": codigo_sala,
        }
        r = supabase.table("salas").insert(nova_sala).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>", methods=["GET"])
@token_required
@professor_required
def detalhes_sala(current_user, sala_id):
    sala = supabase.table("salas").select("*").eq("id", sala_id).single().execute().data
    periodo_filtro = request.args.get("periodo_id")

    missoes = supabase.table("missoes").select("*").eq("sala_id", sala_id).order("ordem").execute().data
    periodos = supabase.table("periodos").select("id, nome").eq("sala_id", sala_id).execute().data

    q = supabase.table("equipes").select("*, periodos(nome)").eq("sala_id", sala_id)
    if periodo_filtro:
        q = q.eq("periodo_id", periodo_filtro)
    equipes = q.execute().data

    alunos_r = (
        supabase.table("aluno_salas")
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
    professor_id = current_user["id"]
    sala = (
        supabase.table("salas")
        .select("id")
        .eq("id", sala_id)
        .eq("professor_id", professor_id)
        .execute()
    )
    if not sala.data:
        return _error("Sala não encontrada ou sem permissão.", 404)
    try:
        missoes_r = supabase.table("missoes").select("id").eq("sala_id", sala_id).execute().data
        missao_ids = [m["id"] for m in missoes_r]
        equipes_r = supabase.table("equipes").select("id").eq("sala_id", sala_id).execute().data
        equipe_ids = [e["id"] for e in equipes_r]
        if equipe_ids:
            supabase.table("equipe_membros").delete().in_("equipe_id", equipe_ids).execute()
        supabase.table("equipes").delete().eq("sala_id", sala_id).execute()
        if missao_ids:
            supabase.table("progresso_missoes").delete().in_("missao_id", missao_ids).execute()
        supabase.table("missoes").delete().eq("sala_id", sala_id).execute()
        supabase.table("periodos").delete().eq("sala_id", sala_id).execute()
        supabase.table("aluno_salas").delete().eq("sala_id", sala_id).execute()
        supabase.table("salas").delete().eq("id", sala_id).execute()
        return _success({"message": "Sala excluída permanentemente."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# PERÍODOS LETIVOS
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/periodos", methods=["GET"])
@token_required
@professor_required
def listar_periodos(current_user, sala_id):
    periodos = (
        supabase.table("periodos")
        .select("id, nome, meta_missoes, criado_em")
        .eq("sala_id", sala_id)
        .order("criado_em")
        .execute().data
    )
    for p in periodos:
        cnt = (
            supabase.table("missoes")
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
    body = request.get_json() or {}
    nome = body.get("nome", "").strip()
    if not nome:
        return _error("O nome do quadrimestre é obrigatório.")
    try:
        meta_missoes = max(1, min(15, int(body.get("meta_missoes", 5))))
    except (ValueError, TypeError):
        meta_missoes = 5
    try:
        r = supabase.table("periodos").insert({
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
    try:
        supabase.table("periodos").delete().eq("id", periodo_id).execute()
        return _success({"message": "Período removido."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# MISSÕES — PROFESSOR (com validações RN-M01/M02)
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/missoes", methods=["POST"])
@token_required
@professor_required
def cadastrar_missao(current_user, sala_id):
    """
    RN-M01: Máximo de 5 missões por período.
    RN-M02: Proibido conflito de número de ordem no mesmo período.
    Retorna HTTP 400 com mensagem JSON (antes era flash() + redirect).
    """
    body = request.get_json() or {}
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título da missão não pode estar vazio.")

    try:
        ordem = int(body.get("ordem", 1))
    except (ValueError, TypeError):
        return _error("Número de ordem inválido.")

    periodo_id = body.get("periodo_id") or None

    # RN-M02: Conflito de ordem
    q_ordem = (
        supabase.table("missoes")
        .select("id")
        .eq("sala_id", sala_id)
        .eq("ordem", ordem)
    )
    q_ordem = q_ordem.eq("periodo_id", periodo_id) if periodo_id else q_ordem.is_("periodo_id", None)
    if q_ordem.execute().data:
        return _error(f"Conflito de Ordem: já existe uma missão com o número de ordem {ordem} neste período.", 400)

    # RN-M01: Limite de 5 por período
    if periodo_id:
        missoes_periodo = (
            supabase.table("missoes")
            .select("id", count="exact")
            .eq("periodo_id", periodo_id)
            .execute()
        )
        if (missoes_periodo.count or 0) >= 5:
            return _error("Limite atingido: este período já possui o máximo de 5 missões.", 400)

    try:
        nova = {
            "sala_id": sala_id,
            "titulo": titulo,
            "descricao": body.get("descricao"),
            "ordem": ordem,
            "xp_reward": int(body.get("xp_reward", 0) or 0),
            "sticker_recompensa_id": body.get("sticker_recompensa_id") or None,
            "data_limite": body.get("data_limite") or None,
            "periodo_id": periodo_id,
            "peso_nota": float(body.get("peso_nota", 1) or 1),
        }
        r = supabase.table("missoes").insert(nova).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/missoes/<missao_id>", methods=["PUT"])
@token_required
@professor_required
def editar_missao(current_user, sala_id, missao_id):
    body = request.get_json() or {}
    try:
        ordem = int(body.get("ordem", 1))
    except (ValueError, TypeError):
        return _error("Número de ordem inválido.")

    periodo_id = body.get("periodo_id") or None

    # RN-M02: Conflito de ordem (excluindo a própria missão)
    q_ordem = (
        supabase.table("missoes")
        .select("id")
        .eq("sala_id", sala_id)
        .eq("ordem", ordem)
        .neq("id", missao_id)
    )
    q_ordem = q_ordem.eq("periodo_id", periodo_id) if periodo_id else q_ordem.is_("periodo_id", None)
    if q_ordem.execute().data:
        return _error(f"Conflito de Ordem: o número {ordem} já está em uso por outra missão.", 400)

    # RN-M01: Verifica limite se a missão está mudando de período
    if periodo_id:
        missao_atual = (
            supabase.table("missoes")
            .select("periodo_id")
            .eq("id", missao_id)
            .single()
            .execute().data
        )
        if missao_atual and missao_atual.get("periodo_id") != periodo_id:
            cnt = (
                supabase.table("missoes")
                .select("id", count="exact")
                .eq("periodo_id", periodo_id)
                .execute()
            )
            if (cnt.count or 0) >= 5:
                return _error("Não é possível mover esta missão: o período de destino já tem 5 missões.", 400)

    try:
        dados = {
            "titulo": body.get("titulo"),
            "descricao": body.get("descricao"),
            "ordem": ordem,
            "xp_reward": int(body.get("xp_reward", 0) or 0),
            "sticker_recompensa_id": body.get("sticker_recompensa_id") or None,
            "data_limite": body.get("data_limite") or None,
            "periodo_id": periodo_id,
            "peso_nota": float(body.get("peso_nota", 1) or 1),
        }
        r = supabase.table("missoes").update(dados).eq("id", missao_id).execute()
        return _success(r.data[0] if r.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/missoes/<missao_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_missao(current_user, sala_id, missao_id):
    try:
        supabase.table("missoes").delete().eq("id", missao_id).execute()
        return _success({"message": "Missão removida da trilha."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# STICKERS — PROFESSOR
# ===========================================================================

@app.route("/api/professor/stickers", methods=["GET"])
@token_required
@professor_required
def listar_stickers(current_user):
    r = supabase.table("stickers").select("*").execute()
    return _success(r.data)


@app.route("/api/professor/stickers", methods=["POST"])
@token_required
@professor_required
def cadastrar_sticker(current_user):
    """Upload de sticker via multipart/form-data."""
    nome = request.form.get("nome", "").strip()
    raridade = request.form.get("raridade", "comum")
    arquivo = request.files.get("imagem_arquivo")

    if not arquivo:
        return _error("Arquivo de imagem é obrigatório.")
    try:
        extensao = arquivo.filename.split(".")[-1]
        nome_arquivo = f"{uuid.uuid4()}.{extensao}"
        supabase.storage.from_("stickers").upload(
            path=nome_arquivo,
            file=arquivo.read(),
            file_options={"content-type": arquivo.content_type},
        )
        imagem_url = supabase.storage.from_("stickers").get_public_url(nome_arquivo)
        r = supabase.table("stickers").insert({
            "nome": nome, "imagem_url": imagem_url, "raridade": raridade
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/stickers/<sticker_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_sticker(current_user, sticker_id):
    try:
        supabase.table("stickers").delete().eq("id", sticker_id).execute()
        return _success({"message": "Sticker removido."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# ALUNOS NA SALA
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/alunos", methods=["POST"])
@token_required
@professor_required
def adicionar_aluno_sala(current_user, sala_id):
    body = request.get_json() or {}
    email_aluno = body.get("email", "").strip().lower()
    try:
        aluno = (
            supabase.table("perfis")
            .select("id, nome")
            .eq("email", email_aluno)
            .eq("role", "aluno")
            .single()
            .execute()
        )
        if not aluno.data:
            return _error("Aluno não encontrado ou não possui conta de aluno.", 404)

        aluno_id = aluno.data["id"]
        ja_tem = supabase.table("aluno_salas").select("sala_id").eq("aluno_id", aluno_id).execute()
        if ja_tem.data:
            return _error(f"O aluno '{aluno.data['nome']}' já está matriculado em outra sala.", 409)

        supabase.table("aluno_salas").insert({"aluno_id": aluno_id, "sala_id": sala_id}).execute()
        return _success({"message": f"{aluno.data['nome']} adicionado à sala!"}, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/alunos/<aluno_id>", methods=["DELETE"])
@token_required
@professor_required
def remover_aluno_sala(current_user, sala_id, aluno_id):
    try:
        supabase.table("aluno_salas").delete().eq("sala_id", sala_id).eq("aluno_id", aluno_id).execute()
        return _success({"message": "Aluno removido da sala."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# EQUIPES
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/equipes", methods=["POST"])
@token_required
@professor_required
def criar_equipe(current_user, sala_id):
    body = request.get_json() or {}
    nome = body.get("nome", "").strip()
    periodo_id = body.get("periodo_id", "").strip()
    if not nome:
        return _error("O nome da equipe não pode ser vazio.")
    if not periodo_id:
        return _error("Selecione um quadrimestre para criar a equipe.")
    try:
        r = supabase.table("equipes").insert({
            "nome": nome, "sala_id": sala_id, "periodo_id": periodo_id
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>", methods=["GET"])
@token_required
@professor_required
def detalhes_equipe(current_user, sala_id, equipe_id):
    equipe = supabase.table("equipes").select("*, periodos(nome)").eq("id", equipe_id).single().execute().data
    membros_r = supabase.table("equipe_membros").select("perfis(id, nome, email)").eq("equipe_id", equipe_id).execute().data
    membros = [m["perfis"] for m in membros_r if m.get("perfis")]

    todos_r = supabase.table("aluno_salas").select("perfis(id, nome)").eq("sala_id", sala_id).execute().data
    todos_alunos = [item["perfis"] for item in todos_r if item.get("perfis")]

    periodo_id_equipe = equipe.get("periodo_id")
    q = supabase.table("equipes").select("id").eq("sala_id", sala_id)
    if periodo_id_equipe:
        q = q.eq("periodo_id", periodo_id_equipe)
    equipes_mesmo_periodo = q.execute().data
    ids_equipes_periodo = [e["id"] for e in equipes_mesmo_periodo]

    ids_em_equipe_periodo = set()
    if ids_equipes_periodo:
        m_r = supabase.table("equipe_membros").select("aluno_id").in_("equipe_id", ids_equipes_periodo).execute().data
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
    try:
        supabase.table("equipe_membros").delete().eq("equipe_id", equipe_id).execute()
        supabase.table("equipes").delete().eq("id", equipe_id).execute()
        return _success({"message": "Equipe removida."})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/membros", methods=["POST"])
@token_required
@professor_required
def adicionar_membro_equipe(current_user, sala_id, equipe_id):
    body = request.get_json() or {}
    aluno_id = body.get("aluno_id")
    try:
        equipe_r = supabase.table("equipes").select("periodo_id").eq("id", equipe_id).single().execute()
        periodo_id_equipe = equipe_r.data.get("periodo_id") if equipe_r.data else None
        if periodo_id_equipe:
            equipes_periodo = (
                supabase.table("equipes")
                .select("id")
                .eq("sala_id", sala_id)
                .eq("periodo_id", periodo_id_equipe)
                .execute().data
            )
            ids_periodo = [e["id"] for e in equipes_periodo]
            if ids_periodo:
                ja_em_equipe = (
                    supabase.table("equipe_membros")
                    .select("equipe_id")
                    .eq("aluno_id", aluno_id)
                    .in_("equipe_id", ids_periodo)
                    .execute().data
                )
                if ja_em_equipe:
                    return _error("Este aluno já pertence a uma equipe neste quadrimestre.", 409)

        supabase.table("equipe_membros").insert({"equipe_id": equipe_id, "aluno_id": aluno_id}).execute()
        return _success({"message": "Aluno adicionado à equipe!"}, 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/equipes/<equipe_id>/membros/<aluno_id>", methods=["DELETE"])
@token_required
@professor_required
def remover_membro_equipe(current_user, sala_id, equipe_id, aluno_id):
    try:
        supabase.table("equipe_membros").delete().eq("equipe_id", equipe_id).eq("aluno_id", aluno_id).execute()
        return _success({"message": "Aluno removido da equipe."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# ENTREGAS — PROFESSOR
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/entregas", methods=["GET"])
@token_required
@professor_required
def listar_entregas(current_user, sala_id):
    filtro_missao = request.args.get("missao_id", "").strip()
    filtro_equipe = request.args.get("equipe_id", "").strip()
    filtro_periodo = request.args.get("periodo_id", "").strip()

    missoes_q = supabase.table("missoes").select("id, titulo, data_limite, periodo_id").eq("sala_id", sala_id)
    if filtro_periodo:
        missoes_q = missoes_q.eq("periodo_id", filtro_periodo)
    if filtro_missao:
        missoes_q = missoes_q.eq("id", filtro_missao)
    missoes_r = missoes_q.order("ordem").execute().data

    todas_missoes_r = supabase.table("missoes").select("id, titulo, periodo_id").eq("sala_id", sala_id).order("ordem").execute().data
    periodos_r = supabase.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data
    equipes_r = supabase.table("equipes").select("id, nome, periodo_id").eq("sala_id", sala_id).order("nome").execute().data

    equipe_ids = [e["id"] for e in equipes_r]
    equipes_obj = {e["id"]: e for e in equipes_r}

    aluno_equipe_por_periodo = {}
    if equipe_ids:
        membros_r = supabase.table("equipe_membros").select("equipe_id, aluno_id").in_("equipe_id", equipe_ids).execute().data
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
        prog_q = (
            supabase.table("progresso_missoes")
            .select("*, perfis(id, nome, email)")
            .in_("missao_id", missao_ids)
            .neq("status", "pendente")
            .order("entregue_em", desc=True)
        )
        progressos = prog_q.execute().data

        if filtro_equipe:
            membros_da_equipe = {
                m["aluno_id"] for m in
                supabase.table("equipe_membros").select("aluno_id").eq("equipe_id", filtro_equipe).execute().data
            }
            progressos = [p for p in progressos if p["aluno_id"] in membros_da_equipe]

        for p in progressos:
            aluno_id_p = p["aluno_id"]
            periodo_da_miss = missao_periodo_map.get(p["missao_id"])
            equipe_obj = aluno_equipe_por_periodo.get((aluno_id_p, periodo_da_miss))
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
    progresso = (
        supabase.table("progresso_missoes")
        .select("*, perfis(id, nome, email), missoes(id, titulo, sala_id, peso_nota, periodo_id)")
        .eq("id", progresso_id)
        .single()
        .execute().data
    )
    sala_id = progresso["missoes"]["sala_id"]
    periodo_id_missao = progresso["missoes"].get("periodo_id")

    equipe = None
    membros_equipe = []
    if periodo_id_missao:
        equipes_r = (
            supabase.table("equipes")
            .select("id, nome")
            .eq("sala_id", sala_id)
            .eq("periodo_id", periodo_id_missao)
            .execute().data
        )
        ids_equipes = [e["id"] for e in equipes_r]
        equipes_map = {e["id"]: e for e in equipes_r}
        if ids_equipes:
            membro_r = (
                supabase.table("equipe_membros")
                .select("equipe_id")
                .eq("aluno_id", progresso["aluno_id"])
                .in_("equipe_id", ids_equipes)
                .execute().data
            )
            if membro_r:
                equipe_id = membro_r[0]["equipe_id"]
                equipe = equipes_map.get(equipe_id)
                membros_r = supabase.table("equipe_membros").select("perfis(nome)").eq("equipe_id", equipe_id).execute().data
                membros_equipe = [m["perfis"]["nome"] for m in membros_r if m.get("perfis")]

    return _success({"progresso": progresso, "equipe": equipe, "membros": membros_equipe, "sala_id": sala_id})


@app.route("/api/professor/entregas/<progresso_id>/corrigir", methods=["POST"])
@token_required
@professor_required
def corrigir_entrega(current_user, progresso_id):
    body = request.get_json() or {}
    try:
        nota = float(body.get("nota", ""))
        if not (0 <= nota <= 10):
            raise ValueError("Nota fora do intervalo 0–10.")
    except (ValueError, TypeError):
        return _error("Nota inválida. Informe um número entre 0 e 10.")

    feedback = body.get("feedback_professor", "").strip()
    try:
        supabase.table("progresso_missoes").update({
            "nota": nota,
            "comentario_professor": feedback,
            "status": "corrigido",
            "validada_professor": True,
        }).eq("id", progresso_id).execute()

        prog_r = (
            supabase.table("progresso_missoes")
            .select("aluno_id, missoes(sticker_recompensa_id)")
            .eq("id", progresso_id)
            .single()
            .execute()
        )
        if prog_r.data:
            aluno_alvo = prog_r.data["aluno_id"]
            sticker_id = prog_r.data["missoes"].get("sticker_recompensa_id")
            if sticker_id:
                ja_tem = (
                    supabase.table("aluno_stickers")
                    .select("sticker_id")
                    .eq("aluno_id", aluno_alvo)
                    .eq("sticker_id", sticker_id)
                    .execute()
                )
                if not ja_tem.data:
                    supabase.table("aluno_stickers").insert({
                        "aluno_id": aluno_alvo, "sticker_id": sticker_id
                    }).execute()

        return _success({"message": "Entrega corrigida e nota lançada!"})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# PAUTA DE NOTAS
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/pauta", methods=["GET"])
@token_required
@professor_required
def pauta(current_user, sala_id):
    periodo_id = request.args.get("periodo_id")
    periodos = supabase.table("periodos").select("id, nome, meta_missoes").eq("sala_id", sala_id).order("criado_em").execute().data

    q_missoes = supabase.table("missoes").select("id, titulo, peso_nota, ordem").eq("sala_id", sala_id)
    if periodo_id:
        q_missoes = q_missoes.eq("periodo_id", periodo_id)
    missoes = q_missoes.order("ordem").execute().data

    q_quiz = supabase.table("quiz").select("id, titulo, periodo_id").eq("sala_id", sala_id)
    if periodo_id:
        q_quiz = q_quiz.eq("periodo_id", periodo_id)
    quizzes = q_quiz.order("criado_em").execute().data

    alunos_r = supabase.table("aluno_salas").select("perfis(id, nome)").eq("sala_id", sala_id).execute().data
    alunos = [item["perfis"] for item in alunos_r if item.get("perfis")]

    missao_ids = [m["id"] for m in missoes]
    progressos = supabase.table("progresso_missoes").select("*").in_("missao_id", missao_ids).execute().data if missao_ids else []

    quiz_ids = [q["id"] for q in quizzes]
    quiz_progressos = (
        supabase.table("quiz_progress")
        .select("aluno_id, quiz_id, score, correct_answers, total_questions, is_late")
        .in_("quiz_id", quiz_ids)
        .execute().data
        if quiz_ids else []
    )
    qprog_idx = {(qp["aluno_id"], qp["quiz_id"]): qp for qp in quiz_progressos}
    aluno_equipe_map = _build_equipe_map_periodo(sala_id, periodo_id)

    pauta_data = []
    for aluno in alunos:
        aid = aluno["id"]
        notas = {}
        soma_m = 0.0
        peso_m = 0.0
        for m in missoes:
            prog = next((p for p in progressos if p["aluno_id"] == aid and p["missao_id"] == m["id"]), None)
            nota = prog["nota"] if prog and prog.get("nota") is not None else None
            notas[m["id"]] = nota
            if nota is not None:
                peso = float(m.get("peso_nota") or 1)
                soma_m += nota * peso
                peso_m += peso
        media_missoes = round(soma_m / peso_m, 2) if peso_m > 0 else None

        quiz_notas = {}
        soma_q = 0.0
        count_q = 0
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


# ===========================================================================
# DESEMPENHO CONSOLIDADO (View)
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/desempenho-consolidado", methods=["GET"])
@token_required
@professor_required
def desempenho_consolidado(current_user, sala_id):
    periodo_id = request.args.get("periodo_id")
    periodos = supabase.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data
    try:
        q = supabase.table("vw_desempenho_consolidado").select("*").eq("sala_id", sala_id)
        if periodo_id:
            missoes_periodo = supabase.table("missoes").select("id").eq("sala_id", sala_id).eq("periodo_id", periodo_id).execute().data
            ids_missoes = [m["id"] for m in missoes_periodo]
            if ids_missoes:
                q = q.in_("missao_id", ids_missoes)
            else:
                return _success({"registros": [], "periodos": periodos})
        registros = q.order("aluno_nome").execute().data
        return _success({"registros": registros, "periodos": periodos})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# DASHBOARD DO ALUNO
# ===========================================================================

@app.route("/api/aluno/dashboard", methods=["GET"])
@token_required
def dashboard_aluno(current_user):
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(aluno_id)

    if not sala_id:
        return _success({"tem_sala": False})

    sala_r = (
        supabase.table("salas")
        .select("id, nome, codigo_acesso, perfis!salas_professor_id_fkey(nome)")
        .eq("id", sala_id)
        .single()
        .execute().data
    )

    periodos_sala = supabase.table("periodos").select("id, nome").eq("sala_id", sala_id).order("criado_em").execute().data
    equipes_sala = supabase.table("equipes").select("id, nome, periodo_id").eq("sala_id", sala_id).execute().data
    equipe_ids_sala = [e["id"] for e in equipes_sala]
    equipes_map = {e["id"]: e for e in equipes_sala}

    equipe = None
    equipe_periodo_nome = None
    periodo_ativo_id = None
    colegas = []

    if equipe_ids_sala:
        membro_r = (
            supabase.table("equipe_membros")
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

                membros_r = supabase.table("equipe_membros").select("perfis(id, nome)").eq("equipe_id", equipe["id"]).execute().data
                colegas = [
                    {"id": m["perfis"]["id"], "nome": m["perfis"]["nome"]}
                    for m in membros_r
                    if m.get("perfis") and m["perfis"]["id"] != aluno_id
                ]

    periodos_trilha, total, concluidas = _build_periodos_trilha(aluno_id, sala_id)

    all_quizzes = (
        supabase.table("quiz")
        .select("id, titulo, dificuldade, xp_reward")
        .eq("sala_id", sala_id)
        .order("criado_em", desc=True)
        .execute().data
    )
    quiz_ids_all = [q["id"] for q in all_quizzes]
    respondidos_ids = set()
    if quiz_ids_all:
        resp_r = (
            supabase.table("quiz_progress")
            .select("quiz_id")
            .eq("aluno_id", aluno_id)
            .in_("quiz_id", quiz_ids_all)
            .execute().data
        )
        respondidos_ids = {r["quiz_id"] for r in resp_r}

    quizzes_disponiveis = [q for q in all_quizzes if q["id"] not in respondidos_ids]

    return _success({
        "tem_sala": True,
        "sala": sala_r,
        "equipe": equipe,
        "equipe_periodo_nome": equipe_periodo_nome,
        "colegas": colegas,
        "periodos_trilha": periodos_trilha,
        "total": total,
        "concluidas": concluidas,
        "quizzes_disponiveis": quizzes_disponiveis,
    })


# ===========================================================================
# MISSÃO — ALUNO
# ===========================================================================

@app.route("/api/aluno/missoes/<missao_id>", methods=["GET"])
@token_required
def ver_missao_aluno(current_user, missao_id):
    aluno_id = current_user["id"]
    missao = (
        supabase.table("missoes")
        .select("*, stickers(imagem_url, nome, raridade), salas(id, nome)")
        .eq("id", missao_id)
        .single()
        .execute().data
    )
    sala_id = missao["salas"]["id"]
    if _get_vinculo_aluno(aluno_id) != sala_id:
        return _error("Acesso negado.", 403)

    periodo_id_missao = missao.get("periodo_id")
    if periodo_id_missao and not _periodo_acessivel(aluno_id, sala_id, periodo_id_missao):
        return _error("Complete todas as missões dos períodos anteriores antes de avançar.", 403)

    prog_r = (
        supabase.table("progresso_missoes")
        .select("*")
        .eq("aluno_id", aluno_id)
        .eq("missao_id", missao_id)
        .execute()
    )
    progresso = prog_r.data[0] if prog_r.data else None

    ant_q = (
        supabase.table("missoes")
        .select("id")
        .eq("sala_id", sala_id)
        .lt("ordem", missao["ordem"])
        .order("ordem", desc=True)
    )
    if periodo_id_missao:
        ant_q = ant_q.eq("periodo_id", periodo_id_missao)
    anteriores = ant_q.limit(1).execute().data
    bloqueada = False
    if anteriores:
        prog_ant = (
            supabase.table("progresso_missoes")
            .select("validada_professor")
            .eq("aluno_id", aluno_id)
            .eq("missao_id", anteriores[0]["id"])
            .execute()
        )
        bloqueada = not (prog_ant.data and prog_ant.data[0]["validada_professor"])

    materiais = (
        supabase.table("biblioteca_materiais")
        .select("id, titulo, descricao, links")
        .eq("missao_id", missao_id)
        .order("criado_em")
        .execute().data
    )

    return _success({
        "missao": missao,
        "progresso": progresso,
        "bloqueada": bloqueada,
        "sala_id": sala_id,
        "materiais": materiais,
    })


@app.route("/api/aluno/missoes/enviar", methods=["POST"])
@token_required
def enviar_missao(current_user):
    aluno_id = current_user["id"]
    body = request.get_json() or {}
    missao_id = body.get("missao_id")

    try:
        missao_r = supabase.table("missoes").select("sala_id, data_limite").eq("id", missao_id).single().execute()
        if _get_vinculo_aluno(aluno_id) != missao_r.data["sala_id"]:
            return _error("Acesso negado.", 403)

        atrasado = False
        data_limite = missao_r.data.get("data_limite")
        if data_limite:
            dl = _parse_dt(data_limite)
            if dl and datetime.now(timezone.utc) > dl:
                atrasado = True

        existente = (
            supabase.table("progresso_missoes")
            .select("id, status")
            .eq("aluno_id", aluno_id)
            .eq("missao_id", missao_id)
            .execute()
        )

        if existente.data:
            if existente.data[0]["status"] == "corrigido":
                return _error("Esta missão já foi corrigida e não pode ser reenviada.", 409)
            supabase.table("progresso_missoes").update({
                "status": "entregue",
                "entregue_em": datetime.now(timezone.utc).isoformat(),
            }).eq("id", existente.data[0]["id"]).execute()
        else:
            supabase.table("progresso_missoes").insert({
                "aluno_id": aluno_id,
                "missao_id": missao_id,
                "status": "entregue",
                "entregue_em": datetime.now(timezone.utc).isoformat(),
            }).execute()

        return _success({"message": "Missão enviada! Aguarde a correção do professor. 🚀", "atrasado": atrasado})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# ENTRADA NA SALA POR CÓDIGO (Aluno)
# ===========================================================================

@app.route("/api/aluno/entrar-sala", methods=["POST"])
@token_required
def entrar_sala_por_codigo(current_user):
    aluno_id = current_user["id"]
    body = request.get_json() or {}
    codigo = body.get("codigo_sala", "").strip().upper()

    try:
        sala = supabase.table("salas").select("id, nome").eq("codigo_acesso", codigo).single().execute()
        if not sala.data:
            return _error("Código inválido. Verifique com seu professor.", 404)

        sala_id = sala.data["id"]
        ja_tem = supabase.table("aluno_salas").select("sala_id").eq("aluno_id", aluno_id).execute()
        if ja_tem.data:
            return _error("Você já está matriculado em uma sala.", 409)

        supabase.table("aluno_salas").insert({"aluno_id": aluno_id, "sala_id": sala_id}).execute()
        return _success({"message": f"Você entrou na sala '{sala.data['nome']}'!", "sala": sala.data}, 201)
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# PERFIS
# ===========================================================================

@app.route("/api/perfil/avatar", methods=["POST"])
@token_required
def upload_avatar(current_user):
    user_id = current_user["id"]
    arquivo = request.files.get("avatar")
    if not arquivo or arquivo.filename == "":
        return _error("Nenhum arquivo selecionado.")

    EXTENSOES_PERMITIDAS = {"jpg", "jpeg", "png", "webp", "gif"}
    extensao = arquivo.filename.rsplit(".", 1)[-1].lower()
    if extensao not in EXTENSOES_PERMITIDAS:
        return _error("Formato inválido. Use JPG, PNG, WebP ou GIF.")

    try:
        nome_arquivo = f"{uuid.uuid4()}.{extensao}"
        supabase.storage.from_("avatars").upload(
            path=nome_arquivo,
            file=arquivo.read(),
            file_options={"content-type": arquivo.content_type},
        )
        avatar_url = supabase.storage.from_("avatars").get_public_url(nome_arquivo)
        supabase.table("perfis").update({"avatar_url": avatar_url}).eq("id", user_id).execute()
        return _success({"avatar_url": avatar_url})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# QUIZZES — RESPOSTAS DO ALUNO (requer validação de backend)
# ===========================================================================

@app.route("/api/aluno/quiz/<quiz_id>/responder", methods=["POST"])
@token_required
def responder_quiz(current_user, quiz_id):
    aluno_id = current_user["id"]
    sala_id = _get_vinculo_aluno(aluno_id)
    if not sala_id:
        return _error("Você não está matriculado em nenhuma sala.", 403)

    quiz = supabase.table("quiz").select("*").eq("id", quiz_id).single().execute().data
    if not quiz or quiz["sala_id"] != sala_id:
        return _error("Quiz não encontrado ou sem acesso.", 404)

    ja_existe = (
        supabase.table("quiz_progress")
        .select("id")
        .eq("aluno_id", aluno_id)
        .eq("quiz_id", quiz_id)
        .execute()
    )
    if ja_existe.data:
        return _error("Você já respondeu este quiz.", 409)

    body = request.get_json() or {}
    respostas = body.get("respostas", {})  # { pergunta_id: "a" | "b" | "c" | "d" }

    perguntas = (
        supabase.table("question")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("ordem")
        .execute().data
    )
    if not perguntas:
        return _error("Este quiz não possui perguntas.", 400)

    correct_answers = sum(
        1 for p in perguntas
        if respostas.get(p["id"], "").strip().lower() == p.get("correct_answer", "").strip().lower()
    )
    total_questions = len(perguntas)
    score = round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0.0

    try:
        supabase.table("quiz_progress").insert({
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


# ===========================================================================
# GEMINI AI — GERAÇÃO DE QUIZ
# ===========================================================================

@app.route("/api/professor/gemini/gerar-quiz", methods=["POST"])
@token_required
@professor_required
def gerar_quiz_gemini(current_user):
    """Gera perguntas de quiz automaticamente usando o Google Gemini."""
    body = request.get_json() or {}
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return _error("Prompt vazio.", 400)

    instrucao = f"""
    Atue como um gerador de quizzes educacionais.
    O usuário pediu: {prompt}

    Retorne APENAS um objeto JSON válido (sem markdown, sem código) com:
    {{
      "title": "string",
      "description": "string",
      "difficulty": "facil",
      "xp_reward": 50,
      "questions": [
        {{
          "question": "string",
          "option_a": "string",
          "option_b": "string",
          "option_c": "string",
          "option_d": "string",
          "correct_answer": "a"
        }}
      ]
    }}
    Dificuldade: 'facil', 'medio' ou 'dificil'. Resposta correta: 'a', 'b', 'c' ou 'd'.
    """
    try:
        resposta = gemini_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=instrucao,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return _success(json.loads(resposta.text))
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# BIBLIOTECA DE MATERIAIS DIDÁTICOS
# ===========================================================================

@app.route("/api/professor/salas/<sala_id>/biblioteca", methods=["GET"])
@token_required
@professor_required
def biblioteca_professor(current_user, sala_id):
    missoes = supabase.table("missoes").select("id, titulo, ordem").eq("sala_id", sala_id).order("ordem").execute().data
    q = supabase.table("biblioteca_materiais").select("*, missoes(titulo)").eq("sala_id", sala_id).order("criado_em", desc=True)
    missao_filtro = request.args.get("missao_id", "")
    if missao_filtro:
        q = q.eq("missao_id", missao_filtro)
    materiais = q.execute().data
    return _success({"missoes": missoes, "materiais": materiais})


@app.route("/api/professor/salas/<sala_id>/biblioteca", methods=["POST"])
@token_required
@professor_required
def criar_material(current_user, sala_id):
    body = request.get_json() or {}
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título do material é obrigatório.")
    try:
        r = supabase.table("biblioteca_materiais").insert({
            "professor_id": current_user["id"],
            "sala_id": sala_id,
            "missao_id": body.get("missao_id") or None,
            "titulo": titulo,
            "descricao": body.get("descricao") or None,
            "links": body.get("links", []),
        }).execute()
        return _success(r.data[0], 201)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/biblioteca/<material_id>", methods=["PUT"])
@token_required
@professor_required
def editar_material(current_user, sala_id, material_id):
    body = request.get_json() or {}
    titulo = body.get("titulo", "").strip()
    if not titulo:
        return _error("O título do material é obrigatório.")
    try:
        r = supabase.table("biblioteca_materiais").update({
            "titulo": titulo,
            "descricao": body.get("descricao") or None,
            "missao_id": body.get("missao_id") or None,
            "links": body.get("links", []),
        }).eq("id", material_id).execute()
        return _success(r.data[0] if r.data else None)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/professor/salas/<sala_id>/biblioteca/<material_id>", methods=["DELETE"])
@token_required
@professor_required
def excluir_material(current_user, sala_id, material_id):
    try:
        supabase.table("biblioteca_materiais").delete().eq("id", material_id).execute()
        return _success({"message": "Material removido."})
    except Exception as e:
        return _error(str(e), 500)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)