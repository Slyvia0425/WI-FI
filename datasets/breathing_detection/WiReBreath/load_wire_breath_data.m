function data = load_wire_breath_data(mat_file)
%LOAD_WIRE_BREATH_DATA Read one WiRe-Breath MAT data file.
%
%   DATA = LOAD_WIRE_BREATH_DATA(MAT_FILE) loads the CSI measurements,
%   ground-truth respiratory signal, timestamps, and public metadata stored
%   in MAT_FILE. This function performs no signal processing or respiratory
%   rate estimation.

    if nargin ~= 1
        error('WiReBreath:InvalidInput', ...
            'Provide the path to one WiRe-Breath MAT file.');
    end

    if isstring(mat_file)
        if ~isscalar(mat_file)
            error('WiReBreath:InvalidInput', 'MAT_FILE must be a scalar path.');
        end
        mat_file = char(mat_file);
    end

    if ~ischar(mat_file) || ~isfile(mat_file)
        error('WiReBreath:FileNotFound', ...
            'The MAT file does not exist: %s', string(mat_file));
    end

    loaded = load(mat_file);
    required_fields = {'csi_complex', 'csi_abs', 'time', ...
        'gt_time', 'gt_signal', 'metadata'};

    for i = 1:numel(required_fields)
        field_name = required_fields{i};
        if ~isfield(loaded, field_name)
            error('WiReBreath:MissingVariable', ...
                'Missing variable "%s" in %s.', field_name, mat_file);
        end
    end

    if size(loaded.csi_complex, 1) ~= numel(loaded.time)
        error('WiReBreath:InvalidDimensions', ...
            'CSI rows must match the number of CSI timestamps.');
    end

    if ~isequal(size(loaded.csi_complex), size(loaded.csi_abs))
        error('WiReBreath:InvalidDimensions', ...
            'CSI complex values and amplitudes must have identical dimensions.');
    end

    if numel(loaded.gt_time) ~= numel(loaded.gt_signal)
        error('WiReBreath:InvalidGroundTruth', ...
            'Ground-truth timestamps and signal must have identical lengths.');
    end

    data = loaded;
end
