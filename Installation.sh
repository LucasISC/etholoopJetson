
cd /home/nvidia/Downloads/

sudo apt -y update
sudo apt -y upgrade

# first get all dependencies
sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm build-essential make cmake cmake-curses-gui g++ git gfortran libjpeg8-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libavutil-dev libxine2-dev libv4l-dev libeigen3-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev  libglew1.6-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libgstreamer1.0 libgstreamer1.0-dev libgstreamer-plugins-bad1.0-0 libgstreamer-plugins-base1.0-0 libgstreamer-plugins-base1.0-dev libgtkglext1 libgtkglext1-dev libcanberra-gtk-module libcanberra-gtk3-module





# Then get XIMEA drivers
sudo apt-get install ca-certificates
wget https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz

tar xzf XIMEA_Linux_SP.tgz
cd package

./install

sudo gpasswd -a "$(whoami)" plugdev
if [ -f /etc/rc.local ]
then
sudo sed -i '/^exit/ d' /etc/rc.local
else
echo '#!/bin/sh -e'                                            | sudo tee    /etc/rc.local > /dev/null
fi
echo 'echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb' | sudo tee -a /etc/rc.local > /dev/null
echo 'exit 0'                                                  | sudo tee -a /etc/rc.local > /dev/null
sudo chmod a+x /etc/rc.local
#enable controlling of memory frequency by user
echo 'KERNEL=="emc_freq_min", ACTION=="add", GROUP="plugdev", MODE="0660"' | sudo tee /etc/udev/rules.d/99-emc_freq.rules > /dev/null
#optional: allow user to use realtime priorities
sudo groupadd -fr realtime
echo '*         - rtprio   0' | sudo tee    /etc/security/limits.d/ximea.conf > /dev/null
echo '@realtime - rtprio  81' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
echo '*         - nice     0' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
echo '@realtime - nice   -16' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
sudo gpasswd -a "$(whoami)" realtime
sudo mkdir /etc/systemd/system/user@.service.d
echo '[Service]'                                                                                 | sudo tee    /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null
echo 'PermissionsStartOnly=true'                                                                 | sudo tee -a /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null
echo 'ExecStartPost=-/bin/sh -c "echo 950000 > /sys/fs/cgroup/cpu/user.slice/cpu.rt_runtime_us"' | sudo tee -a /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null


#OPENCV
cd /home/nvidia/Downloads/

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout "3.4"

In the compiled opencv folder, open CMAKELISTS.txt, search for "WITH_OPENGL" and change OFF to ON.(WITH_OPENGL "Include OpenGL support" OFF)

mkdir build
cd build

!!!!!! CUDA_ARCH_BIN="7.2" for xavier
!!!!!! CUDA_ARCH_BIN="5.3" for nano


cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="" -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D ENABLE_NEON=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D WITH_XIMEA=ON -D WITH_TIFF=OFF -D WITH_OPENGL=ON -D WITH_QT=ON ..
 
make -j4
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig

