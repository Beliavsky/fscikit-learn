program reference_array_resampling_driver
  use sklearn_resampling, only: resample,shuffle
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: cases,c,n,draw,seed,replace_flag,shuffle_flag,kind_,ns,q,typed,i,j,nseed
  integer, allocatable :: requested,si(:,:),before(:),after(:)
  integer(int64), allocatable :: sj(:,:)
  real(dp), allocatable :: sr(:,:)
  character(64), allocatable :: sc(:,:)
  call random_seed(size=nseed);allocate(before(nseed),after(nseed))
  read(*,*) cases
  do c=1,cases
    read(*,*) n,draw,seed,replace_flag,shuffle_flag,kind_,ns,q,typed
    if(draw/=-1) requested=draw
    select case(kind_)
    case(1,2,5,6)
      allocate(si(ns,q));read(*,*) si
    case(3,4)
      allocate(sr(ns,q));read(*,*) sr
    case(7,8)
      allocate(sc(ns,q))
      do j=1,q
        do i=1,ns
          read(*,'(A)') sc(i,j)
        end do
      end do
    case(9,10)
      allocate(sj(ns,q));read(*,*) sj
    end select
    write(*,'(A,1X,I0)') 'BEGIN',c
    select case(kind_)
    case(0);call run()
    case(1);call run(si(:,1))
    case(2);call run(si)
    case(3);call run(sr(:,1))
    case(4);call run(sr)
    case(5);call run(si(:,1)/=0)
    case(6);call run(si/=0)
    case(7);call run(sc(:,1))
    case(8);call run(sc)
    case(9);call run(sj(:,1))
    case(10);call run(sj)
    case(11);block
      type(csr_matrix) :: invalid
      call run(invalid)
    end block
    case(12);call run(1)
    case(13);call run(reshape([1,2],[1,1,2]))
    end select
    write(*,'(A)') 'END'
    if(allocated(si)) deallocate(si)
    if(allocated(sj)) deallocate(sj)
    if(allocated(sr)) deallocate(sr)
    if(allocated(sc)) deallocate(sc)
    if(allocated(requested)) deallocate(requested)
  end do
contains
  subroutine emit_i(name,a)
    character(*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,I0))') name,size(a),a
  end subroutine
  subroutine emit_j(name,a)
    character(*), intent(in) :: name
    integer(int64), intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,I0))') name,size(a),a
  end subroutine
  subroutine emit_r(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,ES26.17E3))') name,size(a),a
  end subroutine
  subroutine emit_c(name,a)
    character(*), intent(in) :: name
    character(*), intent(in) :: a(:)
    integer :: values(size(a)),i
    do i=1,size(a)
      read(a(i),*) values(i)
    end do
    call emit_i(name,values)
  end subroutine
  subroutine run(stratify)
    class(*), optional, intent(in) :: stratify(..)
    integer, allocatable :: ids(:),iv(:),im(:,:),oi(:),om(:,:),ptr(:),columns(:)
    integer(int64), allocatable :: jv(:),jm(:,:),oj(:),ojm(:,:)
    real(dp), allocatable :: rv(:),rm(:,:),orr(:),orm(:,:),data(:)
    logical, allocatable :: bv(:),bm(:,:),ob(:),obm(:,:)
    character(:), allocatable :: cv(:),cm(:,:),oc(:),ocm(:,:)
    type(csr_matrix) :: x,z,saved
    integer :: i,pos
    logical :: replacement
    replacement=replace_flag==1
    call random_seed(get=before)
    if(shuffle_flag==1) then
      ids=shuffle(n,requested,seed)
    else
      ids=resample(n,requested,replacement,seed,stratify)
    end if
    call emit_i('indices',ids)
    if(typed==1) then
      allocate(iv(n),im(n,2),jv(n),jm(n,2),rv(n),rm(n,2),bv(n),bm(n,2),ptr(n+1),columns(4*n),data(4*n))
      allocate(character(24)::cv(n),cm(n,2))
      iv=[(i,i=1,n)];im(:,1)=iv;im(:,2)=-iv
      jv=int(iv,int64)+1099511627776_int64;jm(:,1)=jv;jm(:,2)=-jv
      rv=real(iv,dp)*.125_dp;rm(:,1)=rv;rm(:,2)=-rv
      bv=mod(iv,2)==0;bm(:,1)=bv;bm(:,2)=.not.bv
      ptr(1)=1;pos=1
      do i=1,n
        write(cv(i),'(I0)') i;write(cm(i,1),'(I0)') i;write(cm(i,2),'(I0)') -i
        columns(pos:pos+3)=[2,1,1,3];data(pos:pos+3)=[-real(i,dp),real(i,dp)*.25_dp,real(i,dp)*.75_dp,-0._dp]
        pos=pos+4;ptr(i+1)=pos
      end do
      x=csr_from_arrays(n,3,ptr,columns,data);saved=x
      if(shuffle_flag==1) then
        oi=shuffle(iv,requested,seed);om=shuffle(im,requested,seed)
        oj=shuffle(jv,requested,seed);ojm=shuffle(jm,requested,seed)
        orr=shuffle(rv,requested,seed);orm=shuffle(rm,requested,seed)
        ob=shuffle(bv,requested,seed);obm=shuffle(bm,requested,seed)
        oc=shuffle(cv,requested,seed);ocm=shuffle(cm,requested,seed);z=shuffle(x,requested,seed)
      else
        oi=resample(iv,requested,replacement,seed,stratify);om=resample(im,requested,replacement,seed,stratify)
        oj=resample(jv,requested,replacement,seed,stratify);ojm=resample(jm,requested,replacement,seed,stratify)
        orr=resample(rv,requested,replacement,seed,stratify);orm=resample(rm,requested,replacement,seed,stratify)
        ob=resample(bv,requested,replacement,seed,stratify);obm=resample(bm,requested,replacement,seed,stratify)
        oc=resample(cv,requested,replacement,seed,stratify);ocm=resample(cm,requested,replacement,seed,stratify)
        z=resample(x,requested,replacement,seed,stratify)
      end if
      call emit_i('iv',oi);call emit_i('im',reshape(om,[size(om)]))
      call emit_j('jv',oj);call emit_j('jm',reshape(ojm,[size(ojm)]))
      call emit_r('rv',orr);call emit_r('rm',reshape(orm,[size(orm)]))
      call emit_i('bv',merge(1,0,ob));call emit_i('bm',reshape(merge(1,0,obm),[size(obm)]))
      call emit_c('cv',oc);call emit_c('cm',reshape(ocm,[size(ocm)]));call emit_i('char_length',[len(oc),len(ocm)])
      call emit_i('csr_ptr',z%indptr);call emit_i('csr_indices',z%indices);call emit_r('csr_data',z%data)
      call emit_i('data_preserved',[merge(1,0,all(iv==[(i,i=1,n)]).and.all(im(:,1)==iv).and.all(im(:,2)==-iv).and. &
        all(x%indptr==saved%indptr).and.all(x%indices==saved%indices).and.all(x%data==saved%data))])
    end if
    call random_seed(get=after);call emit_i('rng_preserved',[merge(1,0,all(before==after))])
  end subroutine
end program
