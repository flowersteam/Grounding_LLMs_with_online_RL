#!/bin/bash

SOURCECODE=/home/tcarta/DLP/

DESTINATIONCODE=uez56by@jeanzay:/gpfswork/rech/imi/uez56by/code/DLP



echo Transfer code ...
rsync -azvh -r -e "ssh -i $HOME/.ssh/id_rsa" --exclude-from=exclude_rsync_to_jeanzay.txt $SOURCECODE $DESTINATIONCODE

