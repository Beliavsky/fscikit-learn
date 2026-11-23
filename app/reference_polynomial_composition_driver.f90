program reference_polynomial_composition_driver
  use sklearn_polynomial, only: polynomial_features
  use sklearn_pipeline, only: pipeline
  use sklearn_feature_composition, only: feature_union,column_transformer
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_stateless_preprocessing, only: normalizer
  use sklearn_linear_model, only: ridge
  use sklearn_feature_extraction, only: feature_string
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(polynomial_features) :: poly
  type(normalizer) :: unit
  type(ridge) :: regression
  type(pipeline) :: model
  type(feature_union) :: joined
  type(column_transformer) :: columns
  type(feature_string), allocatable :: names(:)
  real(dp), allocatable :: x(:,:),q(:,:),y(:),weights(:),z(:,:)
  integer :: mode,n,p,m,bias,interaction,named,weighted,i
  character(32) :: number
  read(*,*) mode,n,p,m,poly%min_degree,poly%degree,bias,interaction,named,weighted
  poly%include_bias=bias/=0;poly%interaction_only=interaction/=0;regression%alpha=.75_dp
  allocate(x(n,p),q(m,p),y(n),weights(n),names(p));read(*,*) x;read(*,*) q;read(*,*) y;read(*,*) weights
  do i=1,p
    write(number,'(i0)') i-1;names(i)%text='axis '//trim(number)
  end do
  select case(mode)
  case(0,1)
    allocate(model%steps(2));model%steps(1)%name='poly';allocate(model%steps(1)%model,source=poly)
    if(mode==0) then
      model%steps(2)%name='unit';allocate(model%steps(2)%model,source=unit)
      if(named/=0) then
        z=model%fit_transform(x,feature_names=names)
      else
        z=model%fit_transform(x)
      end if
      call matrix(z);call matrix(model%transform(q));call emit_names(model%get_feature_names_out())
      call integers(reshape(int([model%n_features_in_,model%n_features_out_],int64),[2,1]))
    else
      model%steps(2)%name='ridge';allocate(model%steps(2)%model,source=regression)
      if(weighted/=0) then
        if(named/=0) then
          call model%fit(x,y,weights,names)
        else
          call model%fit(x,y,weights)
        end if
      else if(named/=0) then
        call model%fit(x,y,feature_names=names)
      else
        call model%fit(x,y)
      end if
      call matrix(reshape(model%predict(x),[n,1]));call matrix(reshape(model%predict(q),[m,1]))
      select type(last=>model%steps(2)%model)
      type is(ridge);call matrix(reshape(last%coef_,[size(last%coef_),1]));call matrix(reshape([last%intercept_],[1,1]))
      end select
      select type(first=>model%steps(1)%model)
      class is(dense_transformer);call emit_names(first%feature_names())
      end select
    end if
    call integers(reshape([merge(1_int64,0_int64,model%can_inverse())],[1,1]))
  case(2)
    allocate(joined%transformer_list(2));joined%transformer_list(1)%name='poly'
    allocate(joined%transformer_list(1)%model,source=poly);joined%transformer_list(1)%weight=.7_dp
    joined%transformer_list(2)%name='raw';joined%transformer_list(2)%action='passthrough'
    joined%transformer_list(2)%weight=1.2_dp
    if(named/=0) then
      z=joined%fit_transform(x,names)
    else
      z=joined%fit_transform(x)
    end if
    call matrix(z);call matrix(joined%transform(q));call emit_names(joined%get_feature_names_out())
    call integers(reshape(int([joined%n_features_in_,joined%n_features_out_],int64),[2,1]))
  case(3)
    allocate(columns%transformers(1));columns%transformers(1)%name='poly';columns%transformers(1)%columns=[3,1]
    allocate(columns%transformers(1)%model,source=poly);columns%remainder%action='passthrough'
    if(named/=0) then
      z=columns%fit_transform(x,names)
    else
      z=columns%fit_transform(x)
    end if
    call matrix(z);call matrix(columns%transform(q));call emit_names(columns%get_feature_names_out())
    call integers(reshape(int([columns%n_features_in_,columns%n_features_out_],int64),[2,1]))
  case default;error stop 'unknown polynomial composition mode'
  end select
  call require(poly%n_features_in_==0,'polynomial prototype changed')
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(*(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(*(i0,1x))',a
  end subroutine
  subroutine emit_names(values)
    type(feature_string), intent(in) :: values(:)
    integer(int64), allocatable :: codes(:,:)
    integer :: i,j,width
    width=0
    do i=1,size(values)
      width=max(width,len(values(i)%text))
    end do
    allocate(codes(width,size(values)));codes=-1
    do i=1,size(values)
      do j=1,len(values(i)%text)
        codes(j,i)=iachar(values(i)%text(j:j))
      end do
    end do
    call integers(codes)
  end subroutine
end program
