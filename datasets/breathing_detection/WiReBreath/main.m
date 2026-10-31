% MAIN Load and extract the first WiRe-Breath data file.
%
% Running this script places the CSI data, timestamps, ground truth, and
% metadata from the first Room1 MAT file directly in the MATLAB workspace.

release_root = fileparts(mfilename('fullpath'));
addpath(release_root);
room1_folder = fullfile(release_root, 'data', 'Room1');
files = dir(fullfile(room1_folder, '*.mat'));

if isempty(files)
    error('WiReBreath:NoDataFiles', ...
        'No MAT files were found in %s.', room1_folder);
end

[~, order] = sort({files.name});
data_file = fullfile(files(order(1)).folder, files(order(1)).name);
loaded_data = load_wire_breath_data(data_file);

csi_complex = loaded_data.csi_complex;
csi_abs = loaded_data.csi_abs;
time = loaded_data.time;
gt_time = loaded_data.gt_time;
gt_signal = loaded_data.gt_signal;
metadata = loaded_data.metadata;

fprintf('Loaded: %s\n', data_file);
fprintf('CSI size: %d x %d\n', size(csi_complex, 1), size(csi_complex, 2));
fprintf('Ground-truth points: %d\n', numel(gt_signal));

clear release_root room1_folder files order loaded_data;
