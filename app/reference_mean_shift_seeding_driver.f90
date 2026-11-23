program reference_mean_shift_seeding_driver
  use sklearn_common, only: dp,require
  use sklearn_cluster, only: mean_shift,get_bin_seeds,estimate_bandwidth
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(mean_shift) :: model
  real(dp), allocatable :: x(:,:),q(:,:),starts(:,:),given(:,:),old(:,:),z(:,:)
  real(dp) :: width,quantile,bw,nan
  integer :: cases,c,mode,n,p,nq,freq,iterations,all_points,ns,nsamples,seed,i
  logical :: fallback
  character(64) :: invalid
  nan=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(1,invalid)
  if(len_trim(invalid)>0) then
    call reject(trim(invalid));error stop 'invalid input unexpectedly accepted'
  end if
  read(*,*) cases
  do c=1,cases
    read(*,*) mode,n,p,nq,freq,iterations,all_points,ns,nsamples,seed,width,quantile
    allocate(x(n,p),q(nq,p),given(ns,p))
    do i=1,n
      read(*,*) x(i,:)
    end do
    do i=1,nq
      read(*,*) q(i,:)
    end do
    do i=1,ns
      read(*,*) given(i,:)
    end do
    old=x;write(*,'(a,1x,i0)') 'BEGIN',c
    if(mode==0.or.mode==5) then
      starts=get_bin_seeds(x,width,freq,fallback,float64_width=mode==5)
      call matrix('seeds',starts);call vector('fallback',[real(merge(1,0,fallback),dp)])
    else if(mode==4) then
      if(nsamples==0) then
        bw=estimate_bandwidth(x,quantile,random_state=seed)
      else
        bw=estimate_bandwidth(x,quantile,nsamples,seed)
      end if
      call vector('bandwidth',[bw])
    else
      model=mean_shift();model%bandwidth=width;model%min_bin_freq=freq
      model%max_iter=iterations;model%cluster_all=all_points==1;model%bin_seeding=mode/=3
      bw=width;if(width==-1) bw=estimate_bandwidth(x)
      if(mode==2) then
        starts=given;call model%fit(x,seeds=given)
      else
        if(mode==1) then
          starts=get_bin_seeds(x,bw,freq,float64_width=width==-1)
        else
          starts=x
        end if
        call model%fit(x)
      end if
      call matrix('seeds',starts);call matrix('centers',model%cluster_centers_)
      call vector('labels',real(model%labels_,dp));call vector('prediction',real(model%predict(q),dp))
      call vector('iterations',[real(model%n_iter_,dp)]);call vector('bandwidth',[model%bandwidth_])
      call vector('fallback',[real(merge(1,0,model%binning_failed_),dp)])
    end if
    call matrix('input_preserved',x-old);write(*,'(a)') 'END';flush(6)
    deallocate(x,q,given)
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
    real(dp) :: data(3,2),v
    data=0;data(:,1)=[0._dp,1._dp,3._dp];model=mean_shift()
    select case(name)
    case('negative_bin');z=get_bin_seeds(data,-1._dp)
    case('nan_bin');z=get_bin_seeds(data,nan)
    case('zero_frequency');z=get_bin_seeds(data,1._dp,0)
    case('nan_data');data(1,1)=nan;z=get_bin_seeds(data,1._dp)
    case('bin_overflow');z=get_bin_seeds(data,tiny(1._dp)*.1_dp)
    case('seed_range');data=1.e40_dp;z=get_bin_seeds(data,1._dp)
    case('empty_data');z=get_bin_seeds(data(:0,:),1._dp)
    case('zero_sample');v=estimate_bandwidth(data,n_samples=0)
    case('negative_seed');v=estimate_bandwidth(data,random_state=-1)
    case('nan_quantile');v=estimate_bandwidth(data,nan)
    case('large_quantile');v=estimate_bandwidth(data,1.1_dp)
    case('empty_bins');model%bin_seeding=.true.;model%min_bin_freq=4;model%bandwidth=1;call model%fit(data)
    case('invalid_frequency');model%min_bin_freq=0;call model%fit(data)
    case('nan_bandwidth');model%bandwidth=nan;call model%fit(data)
    case default;error stop 'unknown rejection case'
    end select
  end subroutine
end program
