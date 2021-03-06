function plotCurves(Prediction, Actual, Y)
    
    M = size(Actual,1);
    nof = size(Actual,2);
    X=1:1:M;
    
    for i=2:nof
        switch (i)
            case 2
                label = "Max Temperature";
            case 3
                label = "Min Temperature";
            case 4
                label = "Max DewPoint";
            case 5
                label = "Min DewPoint";
            case 6
                label = "Max Humidity";
            case 7
                label = "Min Humidity";
            case 8
                label = "Max Pressure";
            case 9
                label = "Min Pressure";
            case 10
                label = "Max Visibility";
            case 11
                label = "Min Visibility";
            case 12
                label = "Mean Wind Speed";
            otherwise
                label = "unknown";
        endswitch
        figure;
        Y1=Prediction(:,[i]);
        Y2=Actual(:,[i]);
        plot(X, Y2, '*g', X, Y1, '@r');
        legend('Actual', 'Prediction');
        xlabel('Day');
        ylabel(label);
        filename = sprintf(strcat(label,'.png'));
        saveas(gcf, filename, 'png');
        for k = 1:100
            ;
        endfor
    endfor
    
    Y3 = 0;
    Y4 = 0;
    for i = 1:M
      if Y(:,i) == [1;0;0;0]                                    # Actually ThunderStorm
            Y3 = [Y3,1];
      elseif Y(:,i) == [0;1;0;0]                                # Actually Rainy  
            Y3 = [Y3,2];
      elseif Y(:,i) == [0;0;1;0]                                # Actually Foggy
            Y3 = [Y3,3];                                                      
      else
            Y3 = [Y3,4];                                        # Actually Sunny
      endif
       
      if Prediction(i,[13:16]) == [1,0,0,0]                     # Predicted ThunderStorm
          Y4 = [Y4,1];
      elseif Prediction(i,[13:16]) == [0,1,0,0]                 # Predicted Rainy  
          Y4 = [Y4,2];
      elseif Prediction(i,[13:16]) == [0,0,1,0]                 # Predicted Foggy
          Y4 = [Y4,3];
      else
          Y4 = [Y4,4];                                          # Predicted Sunny
      endif
    endfor
    
    Y3(:,1) = [];
    Y4(:,1) = [];
    figure;     
    plot(X, Y3,"^b",X,Y4,'@r');
    legend('Actual','Prediction');
    xlabel('Day');
    ylabel('Event');  
    filename = sprintf(strcat("Classes",'.png'));
    saveas(gcf, filename, 'png');
    
endfunction