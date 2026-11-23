program reference_regression_datasets_driver
  use sklearn_common, only: dp
  use sklearn_datasets, only: make_low_rank_matrix,make_regression
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  real(dp), allocatable :: x(:,:),y(:,:),weights(:,:),other(:,:),targets(:,:),v(:),coef(:)
  integer :: cases,c,mode,n,p,nt,info,rank,seed,shuf
  real(dp) :: tail,noise,bias,nan
  character(64) :: invalid
  nan=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(1,invalid)
  if(len_trim(invalid)>0) then
    call reject(trim(invalid));error stop 'invalid regression generator input accepted'
  end if
  read(*,*) cases
  do c=1,cases
    read(*,*) mode,n,p,nt,info,rank,seed,shuf,tail,noise,bias
    write(*,'(a,1x,i0)') 'BEGIN',c
    if(mode==0) then
      x=make_low_rank_matrix(n,p,rank,tail,seed);call matrix('x',x)
      other=make_low_rank_matrix(n,p,rank,tail,seed);call matrix('repeated_x',other)
    else if(mode==2) then
      x=make_low_rank_matrix();call matrix('x',x)
    else
      if(rank==0) then
        call make_regression(n,p,x,y,n_informative=info,n_targets=nt,noise=noise,bias=bias,random_state=seed, &
          coef=weights,tail_strength=tail,shuffle=shuf==1)
        call make_regression(n,p,other,targets,n_informative=info,n_targets=nt,noise=noise,bias=bias, &
          random_state=seed,tail_strength=tail,shuffle=shuf==1)
      else
        call make_regression(n,p,x,y,n_informative=info,n_targets=nt,noise=noise,bias=bias,random_state=seed, &
          coef=weights,effective_rank=rank,tail_strength=tail,shuffle=shuf==1)
        call make_regression(n,p,other,targets,n_informative=info,n_targets=nt,noise=noise,bias=bias, &
          random_state=seed,effective_rank=rank,tail_strength=tail,shuffle=shuf==1)
      end if
      call matrix('x',x);call matrix('y',y);call matrix('coef',weights)
      call matrix('without_coef_x',other);call matrix('without_coef_y',targets)
      if(nt==1) then
        if(rank==0) then
          call make_regression(n,p,other,v,n_informative=info,noise=noise,bias=bias,random_state=seed, &
            coef=coef,n_targets=1,tail_strength=tail,shuffle=shuf==1)
        else
          call make_regression(n,p,other,v,n_informative=info,noise=noise,bias=bias,random_state=seed, &
            coef=coef,n_targets=1,effective_rank=rank,tail_strength=tail,shuffle=shuf==1)
        end if
        call matrix('vector_x',other);call vector('vector_y',v);call vector('vector_coef',coef)
      end if
    end if
    write(*,'(a)') 'END';flush(6)
  end do
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do j=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(j)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    select case(name)
    case('zero_samples');x=make_low_rank_matrix(0,2)
    case('zero_features');x=make_low_rank_matrix(2,0)
    case('zero_rank');x=make_low_rank_matrix(2,3,effective_rank=0)
    case('infinite_tail');x=make_low_rank_matrix(tail_strength=ieee_value(0._dp,ieee_positive_inf))
    case('nan_tail');x=make_low_rank_matrix(tail_strength=nan)
    case('large_tail');x=make_low_rank_matrix(tail_strength=1.1_dp)
    case('negative_tail');x=make_low_rank_matrix(tail_strength=-.1_dp)
    case('negative_seed');x=make_low_rank_matrix(random_state=-1)
    case('regression_samples');call make_regression(0,2,x,v)
    case('regression_features');call make_regression(2,0,x,v)
    case('negative_informative');call make_regression(2,3,x,v,n_informative=-1)
    case('zero_targets');call make_regression(2,3,x,y,n_targets=0)
    case('vector_zero_targets');call make_regression(2,3,x,v,n_targets=0)
    case('vector_targets');call make_regression(2,3,x,v,n_targets=2)
    case('negative_noise');call make_regression(2,3,x,v,noise=-1._dp)
    case('nan_noise');call make_regression(2,3,x,v,noise=nan)
    case('infinite_noise');call make_regression(2,3,x,v,noise=ieee_value(0._dp,ieee_positive_inf))
    case('infinite_bias');call make_regression(2,3,x,v,bias=ieee_value(0._dp,ieee_positive_inf))
    case('nan_bias');call make_regression(2,3,x,v,bias=nan)
    case('regression_rank');call make_regression(2,3,x,v,effective_rank=0)
    case('unused_tail');call make_regression(2,3,x,v,tail_strength=2._dp)
    case('regression_seed');call make_regression(2,3,x,v,random_state=-1)
    case default;error stop 'unknown rejection'
    end select
  end subroutine
end program
