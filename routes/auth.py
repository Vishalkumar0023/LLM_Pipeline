from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    make_response,
)
from models import User, db
from utils import (
    get_current_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    set_auth_cookies,
    clear_auth_cookies,
    get_user_folder,
)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    user = get_current_user()
    if user:
        return redirect(url_for("ml.dashboard"))

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if request.is_json:
                resp = jsonify(
                    {
                        "success": True,
                        "redirect": url_for("ml.dashboard"),
                        "access_token": create_access_token(user.id),
                        "refresh_token": create_refresh_token(user.id),
                    }
                )
            else:
                resp = make_response(redirect(url_for("ml.dashboard")))

            set_auth_cookies(resp, user.id)
            return resp

        if request.is_json:
            return jsonify({"error": "Invalid username or password"}), 401

    return render_template("auth.html", mode="login")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page."""
    user = get_current_user()
    if user:
        return redirect(url_for("ml.dashboard"))

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "")

        # SECURITY: Validate inputs before any DB operations
        import re as _re
        validation_errors = []
        if not username or len(username) < 3 or len(username) > 80:
            validation_errors.append("Username must be 3–80 characters")
        elif not _re.match(r"^[a-zA-Z0-9_.\-]+$", username):
            validation_errors.append(
                "Username may only contain letters, numbers, underscores, dots, hyphens"
            )
        if not email or not _re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            validation_errors.append("Invalid email address")
        if not password or len(password) < 8:
            validation_errors.append("Password must be at least 8 characters")
        if validation_errors:
            if request.is_json:
                return jsonify({"error": "; ".join(validation_errors)}), 400
            return render_template("auth.html", mode="signup")

        # Validate uniqueness
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({"error": "Username already exists"}), 400
            return render_template("auth.html", mode="signup")

        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({"error": "Email already registered"}), 400
            return render_template("auth.html", mode="signup")

        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Create user folder
        get_user_folder(user.id)

        if request.is_json:
            resp = jsonify(
                {
                    "success": True,
                    "redirect": url_for("ml.dashboard"),
                    "access_token": create_access_token(user.id),
                    "refresh_token": create_refresh_token(user.id),
                }
            )
        else:
            resp = make_response(redirect(url_for("ml.dashboard")))

        set_auth_cookies(resp, user.id)
        return resp

    return render_template("auth.html", mode="signup")


@auth_bp.route("/logout")
def logout():
    """Logout user — clear JWT cookies."""
    resp = make_response(redirect(url_for("auth.login")))
    clear_auth_cookies(resp)
    return resp


@auth_bp.route("/api/refresh", methods=["POST"])
def refresh_token():
    """
    Refresh the access token using the refresh token.
    Reads from cookie or JSON body.
    """
    token = request.cookies.get("refresh_token")

    if not token and request.is_json:
        token = request.get_json().get("refresh_token")

    if not token:
        return jsonify({"error": "Refresh token required"}), 401

    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return jsonify(
            {"error": "Invalid or expired refresh token", "code": "REFRESH_EXPIRED"}
        ), 401

    user = User.query.get(int(payload["sub"]))
    if not user:
        return jsonify({"error": "User not found"}), 401

    new_access = create_access_token(user.id)

    resp = jsonify({"success": True, "access_token": new_access})
    resp.set_cookie(
        "access_token",
        new_access,
        httponly=True,
        samesite="Lax",
        max_age=int(1800),  # 30 mins
        path="/",
    )
    return resp
