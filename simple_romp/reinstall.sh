pip uninstall simple-romp
python setup.py install
# 刚开始Pip install upgrade simple_romp这个是直接在pypi库发布版本在线下载安装的
# 先卸载pip uninstall simple 再安装 python setup.py install,
# 但是由于现在版本没有提供pypi发布版本的ResultSaver函数所以卸载再安装会出现下面的问题
#ImportError: cannot import name 'ResultSaver' from 'romp' 
# (/home/ssw/.conda/envs/romp/lib/python3.8/site-packages/simple_romp-1.0.5-py3.8-linux-x86_64.egg/romp/__init__.py)
# 解决方法是将/home/ssw/.conda/envs/romp/lib/python3.8/site-packages/romp所有文件复制到simple_romp/romp中，
# 这样python setup.py install 安装的话就不会缺少pypi发布版本的函数了，然后修改simple_romp下面的文件通过reinstall.sh就可以重新安装使用
# 将simple_romp安装到conda环境的好处是可以直接调用,和python命令一样