rng(571,'twister');
nx = 70; ny = 46; episodes = 200; injury_episode = 70;
output_gif = fullfile(pwd,'learning-rerun.gif');
assert(~isfile(output_gif),'Output GIF already exists.');
[x,y] = meshgrid(linspace(0,1,nx),linspace(0,1,ny));
node = reshape(1:numel(x),size(x));

metal = false(ny,nx);
metal(round(0.30*ny):round(0.38*ny),1:3) = true;
metal(round(0.62*ny):round(0.70*ny),1:3) = true;

a = node(:,1:end-1); b = node(:,2:end); pairs_x = [a(:),b(:)];
a = node(1:end-1,:); b = node(2:end,:); pairs_y = [a(:),b(:)];
pairs = [pairs_x;pairs_y];
left_nodes = node(:,1); right_nodes = node(:,end);
fixed = [left_nodes(:);right_nodes(:)];
free = setdiff((1:numel(node))',fixed);

neighbor_kernel = [0 1 0;1 0 1;0 1 0];
upper_field = exp(-x/0.65).*exp(-0.5*((y-0.34)/0.075).^2);
lower_field = exp(-x/0.65).*exp(-0.5*((y-0.66)/0.075).^2);
action_names = ["upper-heavy","balanced","lower-heavy","dark"];
action_weights = [1.00 0.25;0.50 0.50;0.25 1.00;0 0];
q = zeros(1,4);

delivery_history = nan(episodes,1);
stress_history = nan(episodes,1);
reward_history = nan(episodes,1);
action_history = nan(episodes,1);
previous_delivery = NaN; previous_stress = NaN; connected_before = false;

fig = figure('Name','Gray matter learning across 200 trials', ...
    'NumberTitle','off','Color',[0.025 0.03 0.055], ...
    'Position',[45 55 1280 720],'InvertHardcopy','off');
layout = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
microscope = nexttile(layout,[2 1]);
actions_ax = nexttile(layout);
history_ax = nexttile(layout);
sgtitle(layout,{ ...
    'Gray matter learning — 200 consequence-driven trials', ...
    'MATERIAL: normalized conductive phase 1 S/m in 10^{-6} S/m medium  |  DRIVE: 40 Hz sine, 0.50–1.00 V  |  SUBSTRATE: cell-free wet-matter lattice  |  DATE: 2026-11-19'}, ...
    'Color','white','FontSize',14,'FontWeight','bold');

for episode = 1:episodes
    heartbeat = 0.75 + 0.25*sin(2*pi*40*(episode-1)/480);
    fixed_voltage = [heartbeat*ones(numel(left_nodes),1);zeros(numel(right_nodes),1)];
    logits = q/0.20;
    probabilities = exp(logits-max(logits));
    probabilities = probabilities/sum(probabilities);
    action = find(rand <= cumsum(probabilities),1,'first');
    weights = action_weights(action,:);
    optical = heartbeat*(weights(1)*upper_field + weights(2)*lower_field);

    conductivity = 1e-6 + (1-1e-6)*double(metal);
    edge_g = 0.5*(conductivity(pairs(:,1))+conductivity(pairs(:,2)));
    p1 = pairs(:,1); p2 = pairs(:,2);
    laplacian = sparse([p1;p2;p1;p2],[p1;p2;p2;p1], ...
        [edge_g;edge_g;-edge_g;-edge_g],numel(node),numel(node));
    voltage = zeros(numel(node),1);
    voltage(fixed) = fixed_voltage;
    voltage(free) = laplacian(free,free) \ (-laplacian(free,fixed)*fixed_voltage);
    voltage = reshape(voltage,size(metal));
    [gx,gy] = gradient(voltage,1/(nx-1),1/(ny-1));
    jmag = conductivity.*sqrt(gx.^2+gy.^2);
    terminal_g = 0.5*(conductivity(:,end-1)+conductivity(:,end));
    delivery = sum(abs(terminal_g.*(voltage(:,end-1)-voltage(:,end))));
    stress = prctile(jmag(:),99)/(delivery+eps);

    positive = jmag(jmag>0);
    current_scale = prctile(positive,80);
    relative_current = min(jmag/max(current_scale,eps),2);
    neighbor_signal = conv2(relative_current,neighbor_kernel,'same')/4;
    candidates = ~metal & conv2(double(metal),neighbor_kernel,'same')>0;
    growth_probability = min(0.62, ...
        0.20*(0.20+0.80*min(neighbor_signal,1)).*(1+1.6*optical));
    growth = candidates & (rand(size(metal)) < growth_probability);

    connected = any(metal(:,end));
    if isnan(previous_delivery)
        reward = 0;
    else
        delivery_gain = (delivery-previous_delivery)/(abs(previous_delivery)+eps);
        stress_change = (stress-previous_stress)/(abs(previous_stress)+eps);
        reward = 1.6*tanh(delivery_gain) - 0.55*tanh(stress_change) ...
            - 0.10*mean(growth,'all');
        if connected && ~connected_before
            reward = reward + 1.0;
        end
    end
    q(action) = q(action) + 0.18*(reward-q(action));

    delivery_history(episode) = delivery;
    stress_history(episode) = stress;
    reward_history(episode) = reward;
    action_history(episode) = action;

    microscope_image = mat2gray(1-imgaussfilt(double(metal),0.75));
    imagesc(microscope,microscope_image);
    axis(microscope,'image','off'); colormap(microscope,gray(256));
    hold(microscope,'on');
    contour(microscope,optical,[0.35 0.35],'LineColor',[0.15 1 0.55], ...
        'LineWidth',1.4);
    hold(microscope,'off');
    title(microscope,sprintf(['Gray matter microscope view · trial %03d / %03d\n' ...
        'heartbeat %.2f V · whisker action: %s · reward %+0.3f'], ...
        episode,episodes,heartbeat,action_names(action),reward), ...
        'Color','white','FontSize',17,'FontWeight','bold');

    bar(actions_ax,q,'FaceColor',[0.20 0.80 0.70]);
    actions_ax.XTick=1:4; actions_ax.XTickLabel=action_names;
    actions_ax.XTickLabelRotation=18; grid(actions_ax,'on');
    title(actions_ax,'Learned action values','FontWeight','bold');

    yyaxis(history_ax,'left');
    plot(history_ax,1:episode,delivery_history(1:episode), ...
        'Color',[0.08 0.55 0.95],'LineWidth',1.7);
    ylabel(history_ax,'delivered signal');
    yyaxis(history_ax,'right');
    plot(history_ax,1:episode,reward_history(1:episode), ...
        'Color',[0.92 0.35 0.18],'LineWidth',1.2);
    ylabel(history_ax,'reinforcement');
    xlim(history_ax,[1 episodes]); grid(history_ax,'on');
    xline(history_ax,injury_episode,'--','injury','LineWidth',1.3);
    xline(history_ax,100,':','second hundred','LineWidth',1.1);
    title(history_ax,'Delivery and consequence','FontWeight','bold');

    drawnow;
    if episode == 1 || mod(episode,2)==0
        rgb = print(fig,'-RGBImage','-r90');
        [indexed,map] = rgb2ind(rgb,256,'nodither');
        if episode == 1
            imwrite(indexed,map,output_gif,'gif','LoopCount',Inf,'DelayTime',0.075);
        else
            imwrite(indexed,map,output_gif,'gif','WriteMode','append','DelayTime',0.075);
        end
    end
    pause(0.025);

    metal = metal | growth;
    if episode == injury_episode
        cut = round(0.52*nx):round(0.55*nx);
        metal(:,cut) = false;
    end
    previous_delivery = delivery;
    previous_stress = stress;
    connected_before = connected;
end

for hold_frame = 1:10
    rgb = print(fig,'-RGBImage','-r90');
    [indexed,map] = rgb2ind(rgb,256,'nodither');
    imwrite(indexed,map,output_gif,'gif','WriteMode','append','DelayTime',0.075);
end
fprintf('GIF=%s\n',output_gif);
