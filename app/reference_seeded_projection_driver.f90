program reference_seeded_projection_driver
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  use sklearn_random_projection
  implicit none
  class(gaussian_random_projection), allocatable :: model
  real(dp), allocatable :: x(:,:),z(:,:),back(:,:),original(:,:),first(:,:)
  integer, allocatable :: before(:),after(:)
  integer :: cases,c,kind,n,p,k,seed,cache,i,j,ns,mode
  real(dp) :: eps,density
  logical :: sparse_storage
  character(32) :: storage
  call get_command_argument(1,storage);sparse_storage=trim(storage)=='csr-components'
  call random_seed(size=ns);allocate(before(ns),after(ns))
  read(*,*) cases
  do c=1,cases
    read(*,*) kind,n,p,k,seed,cache,eps,density,mode
    allocate(x(n,p));read(*,*) x
    if(kind==0) then
      allocate(gaussian_random_projection::model)
    else
      allocate(sparse_random_projection::model)
      select type(model)
      type is(sparse_random_projection)
        model%density=density;model%store_sparse_components=sparse_storage
      end select
    end if
    model%n_components=k;model%random_state=seed;model%eps=eps;model%compute_inverse_components=cache/=0
    if(mode==7) x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
    if(mode==8) x(1,1)=ieee_value(0._dp,ieee_positive_inf)
    if(mode==1) then
      z=model%transform(x)
    else if(mode==2) then
      z=model%inverse_transform(x)
    else if(mode==4) then
      call model%fit(x);z=model%transform(x(:,:p-1))
    else if(mode==5) then
      call model%fit(x);z=model%inverse_transform(x)
    else if(mode==3) then
      i=johnson_lindenstrauss_min_dim(n,eps);print *,i
    else
      original=x;call random_seed(get=before)
      z=model%fit_transform(x);back=model%inverse_transform(z)
      call random_seed(get=after)
      print *, 'BEGIN',c
      first=component_values();call emit('components',reshape(first,[size(first)]))
      if(sparse_storage) then
        select type(model)
        type is(sparse_random_projection)
          call emit('components_indptr',real(model%components_csr_%indptr,dp))
          call emit('components_indices',real(model%components_csr_%indices,dp))
          call emit('components_data',model%components_csr_%data)
          call emit('component_storage',real([merge(1,0,allocated(model%components_)), &
            merge(1,0,allocated(model%components_csr_%indptr))],dp))
        end select
      end if
      call emit('transform',reshape(z,[size(z)]));call emit('inverse',reshape(back,[size(back)]))
      if(allocated(model%inverse_components_)) then
        call emit('pinv',reshape(model%inverse_components_,[size(model%inverse_components_)]))
      else
        call emit('pinv',[real(dp)::])
      end if
      call emit('metadata',real([model%n_components_,model%n_features_in_,merge(1,0,allocated(model%inverse_components_))],dp))
      select type(model)
      type is(sparse_random_projection);call emit('density',[model%density_])
      class default;call emit('density',[real(dp)::])
      end select
      first=component_values();call model%fit(x)
      call emit('preserved',real([merge(1,0,all(x==original)),merge(1,0,all(before==after)), &
        merge(1,0,all(first==component_values()))],dp))
      model%compute_inverse_components=.false.;call model%fit(x)
      call emit('cleared',[real(merge(1,0,.not.allocated(model%inverse_components_)),dp)])
      print *, 'END',c
    end if
    deallocate(model,x)
  end do
contains
  function component_values() result(a)
    real(dp), allocatable :: a(:,:)
    select type(model)
    type is(sparse_random_projection)
      if(allocated(model%components_csr_%indptr)) then
        a=model%components_csr_%to_dense();return
      end if
    end select
    a=model%components_
  end function
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    write(*,'(*(es25.16e3,1x))') values
  end subroutine
end program
