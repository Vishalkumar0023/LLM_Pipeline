import os
import io
import base64
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import request, jsonify, redirect, url_for, g, current_app
import jwt as pyjwt
from models import User

# Heavy libraries lazy loading
pd = None
np = None
plt = None
DataPipeline = None
ModelTrainer = None
DataCleaner = None


def _load_data_libs():
    global pd, np
    if pd is None:
        import pandas

        pd = pandas
    if np is None:
        import numpy

        np = numpy


def _load_plot_libs():
    global plt
    _load_data_libs()
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot

        plt = matplotlib.pyplot


def _load_pipeline_libs():
    global DataPipeline, ModelTrainer, DataCleaner
    _load_data_libs()
    if DataPipeline is None:
        from data_pipeline import (
            DataPipeline as _DP,
            ModelTrainer as _MT,
            DataCleaner as _DC,
        )

        DataPipeline = _DP
        ModelTrainer = _MT
        DataCleaner = _DC


# JWT Config (loaded from current_app in context)
JWT_ALGORITHM = "HS256"
JWT_ACCESS_EXPIRY = timedelta(minutes=30)
JWT_REFRESH_EXPIRY = timedelta(days=7)


def create_access_token(user_id):
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + JWT_ACCESS_EXPIRY,
    }
    return pyjwt.encode(
        payload, current_app.config["SECRET_KEY"], algorithm=JWT_ALGORITHM
    )


def create_refresh_token(user_id):
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + JWT_REFRESH_EXPIRY,
    }
    return pyjwt.encode(
        payload, current_app.config["SECRET_KEY"], algorithm=JWT_ALGORITHM
    )


def decode_token(token):
    try:
        payload = pyjwt.decode(
            token, current_app.config["SECRET_KEY"], algorithms=[JWT_ALGORITHM]
        )
        return payload
    except pyjwt.ExpiredSignatureError:
        return None
    except pyjwt.InvalidTokenError:
        return None


def get_current_user():
    token = request.cookies.get("access_token")

    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        return None

    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    user = User.query.filter_by(id=int(payload["sub"])).first()
    return user


def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            if request.is_json or request.headers.get("Authorization"):
                return jsonify(
                    {"error": "Authentication required", "code": "TOKEN_EXPIRED"}
                ), 401
            return redirect(url_for("auth.login"))
        g.current_user = user
        return f(*args, **kwargs)

    return decorated


def set_auth_cookies(response, user_id):
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)

    response.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        samesite="Lax",
        max_age=int(JWT_ACCESS_EXPIRY.total_seconds()),
        path="/",
    )
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        samesite="Lax",
        max_age=int(JWT_REFRESH_EXPIRY.total_seconds()),
        path="/",
    )
    return response


def clear_auth_cookies(response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return response


def get_user_folder(user_id):
    BASE_UPLOAD_FOLDER = "user_data"
    folder = os.path.join(BASE_UPLOAD_FOLDER, str(user_id))
    os.makedirs(folder, exist_ok=True)
    return folder


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_plots(df, target_col=None):
    _load_plot_libs()
    plots = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            masked_corr = np.ma.masked_where(mask, corr.values)
            cax = ax.imshow(masked_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            fig.colorbar(cax, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            ax.set_title("Feature Correlation Heatmap")
            plt.tight_layout()
            plots["correlation"] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass

    if numeric_cols:
        try:
            cols_to_plot = numeric_cols[:6]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
            axes = np.atleast_2d(axes).flatten()

            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                data = df[col].dropna()
                ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
                ax.axvline(data.mean(), color="red", linestyle="--", label="Mean")
                ax.axvline(data.median(), color="green", linestyle="--", label="Median")
                ax.set_title(col)
                ax.legend(fontsize=8)

            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Feature Distributions", fontsize=14)
            plt.tight_layout()
            plots["distributions"] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass

    return plots
