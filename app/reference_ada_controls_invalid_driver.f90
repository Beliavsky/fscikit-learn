program reference_ada_controls_invalid_driver
 use sklearn_common,only:dp
 use sklearn_adaboost,only:adaboost_classifier
 use sklearn_adaboost_regression,only:adaboost_regressor
 use sklearn_parameter_space,only:parameter_set,param
 use sklearn_search_adapters,only:set_search_params,fit_search_native
 implicit none
 class(*),allocatable::model
 type(parameter_set)::params
 character(64)::kind,name,token
 real(dp)::x(20,3),y(20),w(20),fraction
 integer::labels(20),i,count
 call get_command_argument(1,kind);call get_command_argument(2,name);call get_command_argument(3,token)
 if(trim(kind)=='regression')then
  allocate(adaboost_regressor::model)
 else
  allocate(adaboost_classifier::model)
 end if
 allocate(params%items(2))
 params%items(1)%name='n_estimators';params%items(1)%value=param(3)
 params%items(2)%name=trim(name)
 if(trim(name)=='sample_weight')then
  params%items(2)%name='learning_rate';params%items(2)%value=param(1._dp)
 else
 if(index(name,'criterion')>0.or.index(name,'splitter')>0.or.trim(name)=='loss'.or. &
    any(trim(token)==[character(64)::'none','sqrt','log2','invalid']))then
  params%items(2)%value=param(trim(token))
 else if(scan(trim(token),'.eEdD')>0.or.any(trim(token)==[character(64)::'nan','inf','-inf']))then
  read(token,*)fraction;params%items(2)%value=param(fraction)
 else
  read(token,*)count;params%items(2)%value=param(count)
 end if
 end if
 call set_search_params(model,params)
 do i=1,20
  x(i,:)=[real(i,dp),real(mod(i,3),dp),sin(real(i,dp))];y(i)=real(i*i,dp)
  labels(i)=merge(-9,8,i<=10)
 end do
 if(trim(name)=='sample_weight')then
  read(token,*)fraction;w=1;w(1)=fraction
  if(trim(token)=='0.0')w=0
  if(trim(kind)=='regression')then
   call fit_search_native(model,x,yr=y,sample_weight=w)
  else
   call fit_search_native(model,x,yi=labels,sample_weight=w)
  end if
 else if(trim(kind)=='regression')then
  call fit_search_native(model,x,yr=y)
 else
  call fit_search_native(model,x,yi=labels)
 end if
 print *,'FIT ACCEPTED'
end program
