program reference_composition_driver
  use reference_composition_models
  implicit none
  type(column_transformer) :: column
  type(feature_union) :: union,prefit
  type(minmax_scaler) :: remainder
  type(standard_scaler) :: scaler
  type(feature_string), allocatable :: input(:),names(:),query_names(:)
  real(dp), allocatable :: x(:,:),q(:,:),fit(:,:),train(:,:),query(:,:),named_query(:,:)
  class(*), allocatable :: model
  integer :: family,n,p,m,kind,config,operation,named,i,j,found
  character(32) :: buffer
  read(*,*) family,n,p,m,kind,config,operation,named
  allocate(x(n,p),q(m,p),input(p));read(*,*) x;read(*,*) q
  do i=1,p
    write(buffer,'(a,i0)') 'field_',i;input(i)%text=trim(buffer)
  end do
  call make_model(kind,config,model)
  if(operation==2) then
    allocate(prefit%transformer_list(1));prefit%transformer_list(1)%name='prefit'
    allocate(prefit%transformer_list(1)%model,source=model)
    if(family==0) then
      call prefit%fit(x(:,[3,1]))
    else
      call prefit%fit(x)
    end if
    deallocate(model);allocate(model,source=prefit%transformer_list(1)%model)
  end if
  selector_calls=0
  if(family==0) then
    allocate(column%transformers(4))
    column%transformers(1)%name='main';allocate(column%transformers(1)%model,source=model)
    column%transformers(1)%weight=-.75_dp
    select case(mod(config,4))
    case(0);column%transformers(1)%columns=[3,1]
    case(1);column%transformers(1)%mask=[.true.,.false.,.true.,.false.,.false.]
    case(2)
      if(named==1) then
        column%transformers(1)%column_names=input([3,1])
      else
        column%transformers(1)%columns=[3,1]
      end if
    case(3);column%transformers(1)%selector=>selector
    end select
    column%transformers(2)%name='raw';column%transformers(2)%action='passthrough'
    column%transformers(2)%columns=[3,1];column%transformers(2)%weight=2
    column%transformers(3)%name='ignored';column%transformers(3)%action='drop';column%transformers(3)%columns=[5]
    column%transformers(4)%name='empty';column%transformers(4)%columns=[integer::]
    allocate(column%transformers(4)%model,source=scaler)
    select case(mod(config,3))
    case(1);column%remainder%action='passthrough'
    case(2)
      column%remainder%action='transform';remainder%feature_min=-1;remainder%feature_max=2
      allocate(column%remainder%model,source=remainder)
    end select
    column%remainder%weight=.5_dp
    if(config==4) then
      column%verbose_feature_names_out=.false.;column%transformers(2)%columns=[4,2]
    end if
    if(config==5) column%name_formatter=>format_name
    if(named==1) then
      if(operation==1) then
        call column%fit(x,input);fit=column%transform(x,input)
      else
        fit=column%fit_transform(x,input)
      end if
    else
      if(operation==1) then
        call column%fit(x);fit=column%transform(x)
      else
        fit=column%fit_transform(x)
      end if
    end if
    train=column%transform(x)
    if(named==1.and.mod(config,2)==1) then
      allocate(named_query(m,5),query_names(5));named_query(:,1)=99;named_query(:,2:5)=q(:,[3,1,4,2])
      query_names(1)%text='new';query_names(2:5)=input([3,1,4,2])
      query=column%transform(named_query,query_names)
    else
      query=column%transform(q)
    end if
    names=column%get_feature_names_out()
    print *,column%n_features_in_,column%n_features_out_,size(column%transformers_),selector_calls
  else
    allocate(union%transformer_list(3))
    union%transformer_list(1)%name='main';allocate(union%transformer_list(1)%model,source=model)
    union%transformer_list(1)%weight=-.75_dp
    union%transformer_list(2)%name='raw';union%transformer_list(2)%action='passthrough'
    union%transformer_list(2)%weight=2
    union%transformer_list(3)%name='ignored';union%transformer_list(3)%action='drop'
    if(config==3.and.any(kind==[9,14,15,17,18,19])) union%verbose_feature_names_out=.false.
    if(named==1) then
      if(operation==1) then
        call union%fit(x,input);fit=union%transform(x)
      else
        fit=union%fit_transform(x,input)
      end if
    else
      if(operation==1) then
        call union%fit(x);fit=union%transform(x)
      else
        fit=union%fit_transform(x)
      end if
    end if
    train=union%transform(x);query=union%transform(q);names=union%get_feature_names_out()
    print *,union%n_features_in_,union%n_features_out_,size(union%transformer_list),selector_calls
  end if
  print *,shape(fit);print '(*(es25.16e3,1x))',fit
  print *,shape(train);print '(*(es25.16e3,1x))',train
  print *,shape(query);print '(*(es25.16e3,1x))',query
  print *,size(names)
  do i=1,size(names)
    print '(a)',names(i)%text
  end do
  if(family==0) then
    print *,5
    do i=1,5
      if(i<=4) then
        found=column%named_transformer_index(column%transformers(i)%name)
      else
        found=column%named_transformer_index('remainder')
      end if
      if(found==0) then
        print *,0,0
      else
        print *,column%transformers_(found)%output_start_-1,column%transformers_(found)%output_stop_
      end if
    end do
  else
    print *,0
  end if
end program
