#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="microlab-site"
SERVICE_FILE="microlab-site.service"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_URL="http://127.0.0.1:8765/"
DRY_RUN=0
SKIP_LINGER=0
SKIP_NGINX=0

usage() {
    cat <<USAGE
Usage: scripts/setup_boot_services.sh [options]

Installs and enables the Microlab Console services needed after reboot.

Options:
  --dry-run          Print commands without running them.
  --skip-linger     Do not run loginctl enable-linger for the current user.
  --skip-nginx      Do not enable/start the nginx system service.
  --project-root    Override the project root path. Defaults to this repo.
  --help            Show this help text.

Run this as the regular rje user, not with sudo. The script calls sudo only for
root-owned system tasks such as loginctl linger and nginx enablement.
USAGE
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-linger)
            SKIP_LINGER=1
            shift
            ;;
        --skip-nginx)
            SKIP_NGINX=1
            shift
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SERVICE_TEMPLATE="$PROJECT_ROOT/ops/systemd/$SERVICE_FILE"
USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
USER_SERVICE_PATH="$USER_SYSTEMD_DIR/$SERVICE_FILE"
CURRENT_USER="$(id -un)"

log() {
    printf '%s\n' "$*"
}

quote_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        quote_command "$@"
    else
        "$@"
    fi
}

run_sudo() {
    if [[ "$EUID" -eq 0 ]]; then
        run "$@"
    else
        run sudo "$@"
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

assert_not_root() {
    if [[ "$EUID" -eq 0 ]]; then
        echo "Run this as the regular project user, not with sudo." >&2
        echo "User systemd services are per-user; sudo would install this for root." >&2
        exit 1
    fi
}

install_user_service() {
    if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
        echo "Missing service template: $SERVICE_TEMPLATE" >&2
        exit 1
    fi

    log "Installing ${SERVICE_NAME}.service into $USER_SYSTEMD_DIR"
    run mkdir -p "$USER_SYSTEMD_DIR"
    run cp "$SERVICE_TEMPLATE" "$USER_SERVICE_PATH"
    run systemctl --user daemon-reload
    run systemctl --user enable --now microlab-site
}

enable_linger() {
    if [[ "$SKIP_LINGER" -eq 1 ]]; then
        log "Skipping user lingering."
        return
    fi

    if loginctl show-user "$CURRENT_USER" -p Linger --value 2>/dev/null | grep -qx "yes"; then
        log "User lingering is already enabled for $CURRENT_USER."
        return
    fi

    log "Enabling user lingering for $CURRENT_USER so user services start after reboot."
    run_sudo loginctl enable-linger "$CURRENT_USER"
}

enable_nginx() {
    if [[ "$SKIP_NGINX" -eq 1 ]]; then
        log "Skipping nginx enablement."
        return
    fi

    if ! command -v nginx >/dev/null 2>&1; then
        log "nginx is not installed; skipping nginx enablement."
        return
    fi

    if systemctl is-enabled --quiet nginx && systemctl is-active --quiet nginx; then
        log "nginx is already enabled and running."
        return
    fi

    log "Enabling and starting nginx."
    run_sudo systemctl enable --now nginx
}

verify() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run complete; no verification performed."
        return
    fi

    systemctl --user is-enabled --quiet "$SERVICE_NAME"
    systemctl --user is-active --quiet "$SERVICE_NAME"
    curl -fsS "$APP_URL" >/dev/null

    if [[ "$SKIP_LINGER" -eq 0 ]]; then
        loginctl show-user "$CURRENT_USER" -p Linger --value | grep -qx "yes"
    fi

    if [[ "$SKIP_NGINX" -eq 0 ]] && command -v nginx >/dev/null 2>&1; then
        systemctl is-enabled --quiet nginx
    fi

    log "Boot services are configured."
}

main() {
    assert_not_root
    require_command systemctl
    require_command loginctl
    require_command curl

    install_user_service
    enable_linger
    enable_nginx
    verify
}

main "$@"
