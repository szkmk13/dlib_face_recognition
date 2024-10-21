# migrate.sh

echo "check dnf version"
dnf --version
echo "check for updates"
dnf check-update
echo "installing cmake"
dnf install -y cmake

echo "upgrading pip"
python3 -m ensurepip --upgrade

echo "installing requirements"
pip3 install -r requirements.txt
python3 manage.py migrate
python3 manage.py collectstatic --noinput