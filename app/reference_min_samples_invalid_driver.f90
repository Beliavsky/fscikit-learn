program reference_min_samples_invalid_driver
 use sklearn_common,only:dp
 use sklearn_tree,only:decision_tree_regressor,decision_tree_classifier,extra_tree_regressor,extra_tree_classifier
 use sklearn_ensemble,only:random_forest_regressor,random_forest_classifier,extra_trees_regressor,extra_trees_classifier, &
  gradient_boosting_regressor,gradient_boosting_classifier
 use sklearn_parameter_space,only:parameter_set,param
 use sklearn_search_adapters,only:set_search_params,fit_search_native
 implicit none
 class(*),allocatable::model
 type(parameter_set)::parameters
 character(64)::argument,name,token
 real(dp)::x(10,1),y(10),fraction
 integer::i,k,count,labels(10)
 call get_command_argument(1,argument);read(argument,*)k
 call get_command_argument(2,name);call get_command_argument(3,token)
 select case(k)
 case(1);allocate(decision_tree_regressor::model)
 case(2);allocate(decision_tree_classifier::model)
 case(3);allocate(extra_tree_regressor::model)
 case(4);allocate(extra_tree_classifier::model)
 case(5);allocate(random_forest_regressor::model)
 case(6);allocate(random_forest_classifier::model)
 case(7);allocate(extra_trees_regressor::model)
 case(8);allocate(extra_trees_classifier::model)
 case(9);allocate(gradient_boosting_regressor::model)
 case(10);allocate(gradient_boosting_classifier::model)
 end select
 allocate(parameters%items(1));parameters%items(1)%name=trim(name)
 if(scan(trim(token),'.eEdD')>0.or.trim(token)=='nan'.or.trim(token)=='inf')then
  read(token,*)fraction;parameters%items(1)%value=param(fraction)
 else
  read(token,*)count;parameters%items(1)%value=param(count)
 end if
 call set_search_params(model,parameters)
 do i=1,10;x(i,1)=real(i,dp);y(i)=real(i*i,dp);labels(i)=mod(i,2);end do
 if(mod(k,2)==0)then
  call fit_search_native(model,x,yi=labels)
 else
  call fit_search_native(model,x,yr=y)
 end if
 print *,'FIT ACCEPTED'
end program
