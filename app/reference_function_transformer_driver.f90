program reference_function_transformer_driver
  use sklearn_function_transformer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_common, only: dp
  use reference_transform_functions
  implicit none
  type(function_transformer) :: model
  type(feature_string), allocatable :: input_names(:),names(:)
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),query(:,:),inverse(:,:)
  integer :: n,p,m,bits,valid,check,naming,operation,stored,j
  character(32) :: buffer
  read(*,*) n,p,m,callback_mode,bits,valid,check,naming,operation,stored
  allocate(x(n,p),q(m,p),input_names(p));read(*,*) x;read(*,*) q
  do j=1,p
    write(buffer,'(a,i0)') 'field_',j;input_names(j)%text=trim(buffer)
  end do
  model%validate=valid==1;model%check_inverse=check==1
  if(btest(bits,0)) model%func=>forward
  if(btest(bits,1)) model%inverse_func=>backward
  if(naming==1.or.naming==3) model%feature_names_out='one-to-one'
  if(naming==2) model%names_func=>names_prefix
  if(operation==2) then
    if(stored==1) then
      z=model%fit_transform(x,input_names)
    else
      z=model%fit_transform(x)
    end if
  else
    if(operation==0) then
      if(stored==1) then
        call model%fit(x,input_names)
      else
        call model%fit(x)
      end if
    end if
    z=model%transform(x)
  end if
  query=model%transform(q);inverse=model%inverse_transform(z)
  print *,model%n_features_in_,merge(1,0,model%inverse_checked_),merge(1,0,model%inverse_consistent_)
  print *,shape(z);print '(*(es25.16e3,1x))',z
  print *,shape(query);print '(*(es25.16e3,1x))',query
  print *,shape(inverse);print '(*(es25.16e3,1x))',inverse
  if(naming>0) then
    if((stored==1.and.operation/=1).or.naming==3) then
      names=model%get_feature_names_out()
    else
      names=model%get_feature_names_out(input_names)
    end if
    print *,size(names)
    do j=1,size(names)
      print '(a)',names(j)%text
    end do
  else
    print *,0
  end if
end program
